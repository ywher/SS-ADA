import logging
from math import cos, pi

def get_scheduler(cfg, len_data, optimizer, start_epoch=0, use_iteration=False, ac_iters=None):
    epochs = (cfg["epochs"] if not use_iteration else 1)
    lr_mode = cfg["lr_scheduler"]["mode"]
    lr_args = cfg["lr_scheduler"]["kwargs"]
    lr_warmup = cfg["lr_scheduler"]["warmup"]
    if lr_warmup:
        warmup_mode = cfg["lr_scheduler"]["warmup_mode"]
        warmup_iters = cfg["lr_scheduler"]["warmup_iters"]
        warmup_ratio = cfg["lr_scheduler"].get("warmup_ratio", 5e-4)
        # print('len_data', len_data)
        # assert warmup_iters < len_data, "warmup_iters should be less than len_data"
        lr_scheduler = WarmupLRScheduler(
            lr_mode, 
            lr_args, 
            len_data, 
            optimizer, 
            epochs, 
            start_epoch, 
            warmup_iters, 
            warmup_mode, 
            warmup_ratio, 
            ac_iters,
            )
    else:
        lr_scheduler = LRScheduler(
            lr_mode, 
            lr_args, 
            len_data, 
            optimizer, 
            epochs, 
            start_epoch,
            ac_iters,
        )
    return lr_scheduler


class LRScheduler(object):
    """
    Learning rate scheduler for optimizing the learning rate during training.

    Args:
        mode (str): The mode of the learning rate scheduler. Must be one of ["multistep", "poly", "cosine"].
        lr_args (dict): Additional arguments specific to the chosen mode.
        data_size (int): The size of the training data.
        optimizer (torch.optim.Optimizer): The optimizer to adjust the learning rate for.
        num_epochs (int): The total number of epochs for training.
        start_epochs (int): The starting epoch for training.

    Attributes:
        mode (str): The mode of the learning rate scheduler.
        optimizer (torch.optim.Optimizer): The optimizer to adjust the learning rate for.
        data_size (int): The size of the training data.
        cur_iter (int): The current iteration.
        max_iter (int): The maximum number of iterations.
        base_lr (list): The base learning rate for each parameter group in the optimizer.
        cur_lr (list): The current learning rate for each parameter group.

    Methods:
        step(): Performs a step in the learning rate scheduler.
        _step(): Performs the specific step based on the chosen mode.
        get_lr(): Returns the current learning rate.
        update_lr(): Updates the learning rate in the optimizer.

    Raises:
        NotImplementedError: If the chosen mode is not implemented.

    """

    def __init__(self, mode, lr_args, data_size, optimizer, num_epochs, start_epochs, ac_iters=None):
        super(LRScheduler, self).__init__()
        logger = logging.getLogger("global")

        assert mode in ["multistep", "poly", "cosine"]
        self.mode = mode
        self.optimizer = optimizer
        self.data_size = data_size

        self.cur_iter = start_epochs * data_size
        self.max_iter = num_epochs * data_size

        # set learning rate
        self.base_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.cur_lr = [lr for lr in self.base_lr]

        # poly kwargs
        # TODO
        if mode == "poly":
            self.power = lr_args["power"] if lr_args.get("power", False) else 0.9
            logger.info("The kwargs for lr scheduler: {}".format(self.power))
        if mode == "milestones":
            default_mist = list(range(0, num_epochs, num_epochs // 3))[1:]
            self.milestones = (
                lr_args["milestones"]
                if lr_args.get("milestones", False)
                else default_mist
            )
            logger.info("The kwargs for lr scheduler: {}".format(self.milestones))
        if mode == "cosine":
            self.targetlr = lr_args["targetlr"]
            logger.info("The kwargs for lr scheduler: {}".format(self.targetlr))

    def step(self):
        """
        Performs a step in the learning rate scheduler.
        """
        self._step()
        self.update_lr()
        self.cur_iter += 1

    def _step(self):
        """
        Performs the specific step based on the chosen mode.
        """
        if self.mode == "step":
            epoch = self.cur_iter // self.data_size
            power = sum([1 for s in self.milestones if s <= epoch])
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * pow(0.1, power)
                self.cur_lr[i] = adj_lr
        elif self.mode == "poly":
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * (
                    (1 - float(self.cur_iter) / self.max_iter) ** (self.power)
                )
                self.cur_lr[i] = adj_lr
        elif self.mode == "cosine":
            for i, lr in enumerate(self.base_lr):
                adj_lr = (
                    self.targetlr
                    + (lr - self.targetlr)
                    * (1 + cos(pi * self.cur_iter / self.max_iter))
                    / 2
                )
                self.cur_lr[i] = adj_lr
        else:
            raise NotImplementedError

    def get_lr(self):
        """
        Returns the current learning rate.

        Returns:
            list: The current learning rate for each parameter group.

        """
        return self.cur_lr

    def update_lr(self):
        """
        Updates the learning rate in the optimizer.
        """
        for param_group, lr in zip(self.optimizer.param_groups, self.cur_lr):
            param_group["lr"] = lr


class WarmupLRScheduler(object):
    """
    A learning rate scheduler with warmup for optimization algorithms.

    Args:
        mode (str): The mode of the learning rate scheduler. Supported modes are "multistep", "poly", and "cosine".
        lr_args (dict): Additional arguments for the learning rate scheduler.
        data_size (int): The size of the training data.
        optimizer (torch.optim.Optimizer): The optimizer to adjust the learning rate for.
        num_epochs (int): The total number of epochs for training.
        start_epochs (int): The starting epoch for training.
        warmup_iters (int): The number of warmup iterations.

    Raises:
        NotImplementedError: If an invalid mode is provided.

    """

    def __init__(self, mode, lr_args, data_size, optimizer, num_epochs, start_epochs, warmup_iters, warmup_mode, warmup_ratio, ac_iters=None):
        super(WarmupLRScheduler, self).__init__()
        logger = logging.getLogger("global")

        assert mode in ["multistep", "poly", "cosine"]
        self.mode = mode
        self.optimizer = optimizer
        self.data_size = data_size

        self.cur_iter = start_epochs * data_size
        self.max_iter = num_epochs * data_size
        self.ac_iters = ac_iters
        if self.ac_iters is not None:
            self.cur_iter = self.ac_iters.get_cur_epoch_init_iters(start_epochs)
            self.max_iter = self.ac_iters.total_iters
        self.warmup_iters = warmup_iters
        self.warmup_mode = warmup_mode
        assert warmup_mode in ["linear", "exp"]
        self.warmup_ratio = warmup_ratio

        # set learning rate
        self.base_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.cur_lr = [lr for lr in self.base_lr]

        # poly kwargs
        if mode == "poly":
            self.power = lr_args["power"] if lr_args.get("power", False) else 0.9
            logger.info("The kwargs for lr scheduler: {}".format(self.power))
        elif mode == "milestones":
            default_mist = list(range(0, num_epochs, num_epochs // 3))[1:]
            self.milestones = (
                lr_args["milestones"]
                if lr_args.get("milestones", False)
                else default_mist
            )
            logger.info("The kwargs for lr scheduler: {}".format(self.milestones))
        elif mode == "cosine":
            self.targetlr = lr_args["targetlr"]
            logger.info("The kwargs for lr scheduler: {}".format(self.targetlr))
        
        if self.warmup_mode == "linear":
            logger.info("Using linear warmup.")
        elif self.warmup_mode == "exp":
            logger.info("Using exponential warmup.")
        else:
            raise NotImplementedError("Invalid mode for warmup.")

    def step(self):
        """
        Perform a single optimization step.

        """
        self._step()
        self.update_lr()
        self.cur_iter += 1

    def _step(self):
        """
        Perform the actual optimization step based on the current iteration.

        """
        if self.cur_iter < self.warmup_iters:
            # Warmup phase
            if self.warmup_mode == "linear":
                # Linear warmup
                for i, lr in enumerate(self.base_lr):
                    adj_lr = lr * (float(self.cur_iter) / self.warmup_iters)
                    self.cur_lr[i] = adj_lr
            elif self.warmup_mode == "exp":
                # Exponential warmup, 
                # TODO: check if this is correct
                for i, lr in enumerate(self.base_lr):
                    adj_lr = lr * self.warmup_ratio ** (1 - self.cur_iter / self.warmup_iters)
                    self.cur_lr[i] = adj_lr
        else:
            if self.mode == "step":
                epoch = (self.cur_iter - self.warmup_iters) // self.data_size
                power = sum([1 for s in self.milestones if s <= epoch])
                for i, lr in enumerate(self.base_lr):
                    adj_lr = lr * pow(0.1, power)
                    self.cur_lr[i] = adj_lr
            elif self.mode == "poly":
                for i, lr in enumerate(self.base_lr):
                    adj_lr = lr * (
                        (1 - float(self.cur_iter - self.warmup_iters) / (self.max_iter - self.warmup_iters)) ** self.power
                    )
                    self.cur_lr[i] = adj_lr
            elif self.mode == "cosine":
                for i, lr in enumerate(self.base_lr):
                    adj_lr = (
                        self.targetlr
                        + (lr - self.targetlr)
                        * (1 + cos(pi * (self.cur_iter - self.warmup_iters) / (self.max_iter - self.warmup_iters)))
                        / 2
                    )
                    self.cur_lr[i] = adj_lr
            else:
                raise NotImplementedError("Invalid mode for scheduler")

    def get_lr(self):
        """
        Get the current learning rate.

        Returns:
            list: The current learning rate for each parameter group in the optimizer.

        """
        return self.cur_lr

    def update_lr(self):
        """
        Update the learning rate in the optimizer.

        """
        for param_group, lr in zip(self.optimizer.param_groups, self.cur_lr):
            param_group["lr"] = lr