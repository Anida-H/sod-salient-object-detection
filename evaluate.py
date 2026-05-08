import torch
import numpy as np
import matplotlib.pyplot as plt

from sod_model import MiniUNet
from data_loader import create_dataloader


# -----------------------------------
# Metrics
# -----------------------------------
def calculate_metrics(preds, masks, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    masks = (masks > threshold).float()

    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    tp = (preds_flat * masks_flat).sum()
    fp = (preds_flat * (1 - masks_flat)).sum()
    fn = ((1 - preds_flat) * masks_flat).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)

    intersection = (preds_flat * masks_flat).sum()
    union = preds_flat.sum() + masks_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return (
        precision.item(),
        recall.item(),
        f1.item(),
        iou.item()
    )


# -----------------------------------
# Evaluation Function
# -----------------------------------
def evaluate_model(
    test_image_dir,
    test_mask_dir,
    model_path="best_mini_unet_sod_model.pth",
    batch_size=8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = create_dataloader(
        test_image_dir,
        test_mask_dir,
        batch_size=batch_size,
        shuffle=False
    )

    model = MiniUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            precision, recall, f1, iou = calculate_metrics(outputs, masks)

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_iou += iou

    num_batches = len(test_loader)

    print("Evaluation Results:")
    print(f"Precision: {total_precision / num_batches:.4f}")
    print(f"Recall: {total_recall / num_batches:.4f}")
    print(f"F1-score: {total_f1 / num_batches:.4f}")
    print(f"IoU: {total_iou / num_batches:.4f}")


# -----------------------------------
# Visualization Function
# -----------------------------------
def visualize_prediction(
    image,
    ground_truth,
    prediction
):
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float()

    image = image.permute(1, 2, 0).cpu().numpy()
    ground_truth = ground_truth.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()

    overlay = image.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], prediction)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()


# -----------------------------------
# Example usage
# -----------------------------------
if __name__ == "__main__":
    evaluate_model(
        test_image_dir="dataset/test/images",
        test_mask_dir="dataset/test/masks",
        model_path="best_mini_unet_sod_model.pth",
        batch_size=8
    )