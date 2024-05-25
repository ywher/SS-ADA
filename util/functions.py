import torch
import os
import numpy as np
import cv2
cv2.setNumThreads(0)

def calculate_entropy(predictions):
    """
    Calculate the normalized entropy of predictions in pixel level.

    Args:
        predictions (torch.Tensor): Predictions tensor of shape (batch_size, num_classes, height, width) or (num_classes, height, width).

    Returns:
        torch.Tensor: Entropy tensor of shape (batch_size, height, width).
    """
    if len(predictions.size()) == 4:
        batch_size, num_classes, height, width = predictions.size()
        entropy = torch.zeros(batch_size, height, width)
        for i in range(batch_size):
            pred_prob = torch.softmax(predictions[i], dim=0)  # Compute probabilities
            entropy[i] = torch.sum(-pred_prob * torch.log2(pred_prob + 1e-10), dim=0) / np.log2(num_classes)  # Compute entropy
    elif len(predictions.size()) == 3:
        num_classes, height, width = predictions.size()
        pred_prob = torch.softmax(predictions, dim=0)
        # Compute entropy, normalize to [0, 1]
        entropy = torch.sum(-pred_prob * torch.log2(pred_prob + 1e-10), dim=0) / np.log2(num_classes)
        
    return entropy

def calculate_confidence(predictions):
    """
    Calculate the confidence of predictions in pixel level.

    Args:
        predictions (torch.Tensor): Predictions tensor of shape (batch_size, num_classes, height, width) or (num_classes, height, width).

    Returns:
        torch.Tensor: Confidence tensor of shape (batch_size, height, width).
    """
    if len(predictions.size()) == 4:
        batch_size, _, height, width = predictions.size()
        confidence = torch.zeros(batch_size, height, width)
        for i in range(batch_size):
            pred_prob = torch.softmax(predictions[i], dim=0)  # Compute probabilities
            max_confidence, _ = torch.max(pred_prob, dim=0)  # Compute max confidence along channel dimension
            confidence[i] = max_confidence
    elif len(predictions.size()) == 3:
        _, height, width = predictions.size()
        pred_prob = torch.softmax(predictions, dim=0)
        max_confidence, _ = torch.max(pred_prob, dim=0)  # Compute max confidence along channel dimension
        confidence = max_confidence
        
    return confidence

def save_prediction_results(pred_dict, output_dir, return_pred=False):
    """
    Save prediction results.

    Args:
        pred_dict dict: Dictionary containing predictions. {img_name: pred (C, H, W) before softmax}
        output_dir (str): Output directory to save prediction results.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # bar = tqdm(total=len(pred_dict))
    for image_name, pred in pred_dict.items():
        save_path = os.path.join(output_dir, image_name)
        if len(pred.size()) == 4:
            pred = pred.squeeze(0)
        pred = torch.softmax(pred, dim=0)  # Apply softmax, (C, H, W)
        pred = torch.argmax(pred, dim=0)  # Get the class index with highest probability, (H, W)
        cv2.imwrite(save_path, pred.cpu().numpy().astype(np.uint8))
        if return_pred:
            assert len(pred.size()) == 2, "Prediction should be in shape (H, W)"
            return pred
        
        # bar.update(1)
    # bar.close()

def save_entropy_results(entropy_dict, output_dir):
    """
    Save entropy results.

    Args:
        entropy_dict dict: Dictionary containing entropy values. {img_name: entropy in pixel level}
        output_dir (str): Output directory to save entropy results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Save entropy predictions
    # entropy_img_dict = {}
    # bar = tqdm(total=len(entropy_dict))
    # with open(os.path.join(output_dir, '..', 'entropy.csv'), 'w') as f:
    #     f.write("Image Name,Entropy\n")
    for img_name, entropy in entropy_dict.items():
        entropy_np = entropy.cpu().numpy()
        # image_entropy = np.mean(entropy_np)
        # entropy_img_dict[img_name] = image_entropy
        # f.write(f"{img_name},{image_entropy}\n")
        entropy_path = os.path.join(output_dir, img_name)
        entropy_img = (entropy_np * 255).astype(np.uint8).clip(0, 255)
        cv2.imwrite(entropy_path, entropy_img)
            # bar.update(1)
    # f.close()
    # bar.close()
    return entropy_np

def save_confidence_results(confidence_dict, output_dir):
    """
    Save confidence results.

    Args:
        confidence_dict (dict): Dictionary containing confidence values. {img_name: confidence in pixel level}
        output_dir (str): Output directory to save confidence results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    for img_name, confidence in confidence_dict.items():
        confidence_np = confidence.cpu().numpy()
        confidence_path = os.path.join(output_dir, img_name)
        confidence_img = (confidence_np * 255).astype(np.uint8).clip(0, 255)
        cv2.imwrite(confidence_path, confidence_img)
        
    return confidence_np


def generate_confidence_mask(pred_mask, pred_confi, iou_thresholds, threshold_ratio=1.0):
    '''
    pred_mask:  [B,H,W] or [H,W]
    pred_confi: [B,H,W] or [H,W]
    iou_thresholds: 1 * C, torch.tensor (gpu)
    '''
    assert pred_mask.size() == pred_confi.size(), "the size of pred_mask and pred_confi should be the same"

    # 根据类别索引获取对应的iou阈值
    if iou_thresholds.device.type == 'cpu':
        iou_thresholds_tensor = iou_thresholds.clone().to(pred_mask.device)
    else:
        iou_thresholds_tensor = iou_thresholds.clone()
        
    iou_thresholds_tensor = iou_thresholds_tensor * threshold_ratio

    # 计算每个像素点的预测置信度是否大于等于对应类别的iou阈值
    mask = pred_confi >= iou_thresholds_tensor[pred_mask]

    return mask

if __name__ == '__main__':
    # 示例用法
    H, W = 5, 5
    num_class = 4  
    iou_thresholds = [0.5, 0.6, 0.7, 0.8]  # 示例iou阈值数组
    iou_thresholds = torch.tensor(iou_thresholds)
    pred = torch.randn(num_class, H, W, device='cuda')  # 示例预测结果张量
    pred_confi = torch.rand(H, W, device='cuda')  # 示例预测置信度张量
    confidence_mask = generate_confidence_mask(pred, pred_confi, iou_thresholds)