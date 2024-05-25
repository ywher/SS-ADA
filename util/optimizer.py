# import torch
import torch.optim as optim

def set_optimizer_bisenet(model, cfg_optim):
    optim_type = cfg_optim["type"]
    optim_times = cfg_optim["times"]
    optim_kwargs = cfg_optim["kwargs"]
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': optim_kwargs['lr'] * optim_times},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': optim_kwargs['lr'] * optim_times},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    if optim_type == "SGD":
        optimizer = optim.SGD(
            params_list,
            **optim_kwargs,
        )
    elif optim_type == "adam":
        optimizer = optim.Adam(
            params_list,
            **optim_kwargs,
        )
    else:
        optimizer = None
    assert optimizer is not None, "optimizer type is not supported by LightSeg"
    return optimizer