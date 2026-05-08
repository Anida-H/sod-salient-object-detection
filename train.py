import torch
import torch.nn as nn
import torch.optim as optim

from sod_model import MiniUNet
from data_loader import create_dataloader


# -----------------------------------
# IoU Loss
# -----------------------------------
def iou_loss(preds, masks, smooth=1e-6):
    preds = torch.sigmoid(preds)

    preds = preds.view(-1)
    masks = masks.view(-1)

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return 1 - iou


# -----------------------------------
# Combined BCE + IoU Loss
# -----------------------------------
def combined_loss(preds, masks):
    bce = nn.BCEWithLogitsLoss()(preds, masks)
    iou = iou_loss(preds, masks)

    return bce + 0.5 * iou


# -----------------------------------
# Training Function
# -----------------------------------
def train_model(
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    epochs=25,
    batch_size=8,
    learning_rate=1e-3,
    save_path="best_mini_unet_sod_model.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(
        train_image_dir,
        train_mask_dir,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = create_dataloader(
        val_image_dir,
        val_mask_dir,
        batch_size=batch_size,
        shuffle=False
    )

    model = MiniUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = combined_loss(outputs, masks)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

    print("Training completed.")


# -----------------------------------
# Example usage
# -----------------------------------
if __name__ == "__main__":
    train_model(
        train_image_dir="dataset/train/images",
        train_mask_dir="dataset/train/masks",
        val_image_dir="dataset/val/images",
        val_mask_dir="dataset/val/masks",
        epochs=25,
        batch_size=8,
        learning_rate=1e-3
    )