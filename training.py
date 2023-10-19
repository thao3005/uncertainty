import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from mcdropout import MCDropout3D
from preprocess import Dataset2D, Dataset3D, NormalizeIntensity
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from MyUNet import UNet3D, DeepUNet2D
# from MonaiUNet import UNet
from monai.losses import DiceCELoss
# from torchmetrics.classification import Dice
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocess import train_loader, valid_loader, train_image_paths, train_label_paths, valid_image_paths, valid_label_paths

# Constants
NUM_EPOCHS = 5
MODE = '2d'
DEVICE = torch.device('cuda')
BEST_MODEL_PATH = None
TRAINING_RESULTS_DIR = 'training_results'

# ================= Loss Functions and Dice Score =================

def dice_coefficient(target, preds):
    temp = torch.zeros_like(preds[:, 1])
    temp[temp > 0.5] = 1
    preds = temp.unsqueeze(1)
    intersection = (preds * target).sum().float()
    set_sum = preds.sum() + target.sum()
    
    dice = (2 * intersection + 1e-8) / (set_sum + 1e-8) 
    
    return dice

dice_ce_loss = DiceCELoss(to_onehot_y=True)

# ================= Training Utilities =================

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for batch_images, batch_labels in tqdm(loader, desc="Training", ncols=100):
        batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(batch_images).to(DEVICE)
        loss = dice_ce_loss(outputs, batch_labels)
        
        loss.backward()
        optimizer.step()

        # threshold_outputs = (outputs >= 0.5).float()
        dice_value = dice_coefficient(batch_labels, outputs)
        running_dice += dice_value.item()

        running_loss += loss.item()

    return running_loss / len(loader), running_dice / len(loader)

def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for batch_images, batch_labels in tqdm(loader, desc="Validation", ncols=100):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)
            loss = dice_ce_loss(outputs, batch_labels)

            dice_value = dice_coefficient(batch_labels, outputs)
            running_dice += dice_value.item()

            running_loss += loss.item()

    return running_loss / len(loader), running_dice / len(loader)

def create_training_directory():
    if not os.path.exists(TRAINING_RESULTS_DIR):
        os.mkdir(TRAINING_RESULTS_DIR)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(TRAINING_RESULTS_DIR, current_time)
    os.mkdir(path)
    return path

def log_to_file(path, message):
    with open(path, 'a') as log_file:
        log_file.write(message + '\n')

def main():
    training_path = create_training_directory()
    best_model_path = os.path.join(training_path, 'best_model.pth')
    log_path = os.path.join(training_path, 'training_log.txt')

    mode_print = f"Running {MODE} model"
    print(mode_print)
    log_to_file(log_path, mode_print)

    device_print = f"Running {DEVICE} model"
    print(device_print)
    log_to_file(log_path, device_print)
    
    if MODE == '2d':
        model = DeepUNet2D().to(DEVICE)
    elif MODE == '3d':
        model = UNet3D().to(DEVICE)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded weights from best_model.pth")

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    train_losses, train_dices, val_losses, val_dices = [], [], [], []
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        epoch_msg = f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
        print(epoch_msg)
        log_to_file(log_path, epoch_msg)

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_dice = evaluate(model, valid_loader)

        training_results_msg = f"\nTraining - Loss: {train_loss:.4f}, Dice Score: {train_dice:.4f}"
        validation_results_msg = f"Validation - Loss: {val_loss:.4f}, Dice Score: {val_dice:.4f}"
        
        print(training_results_msg)
        print(validation_results_msg)

        log_to_file(log_path, training_results_msg)
        log_to_file(log_path, validation_results_msg)

        train_losses.append(train_loss)
        train_dices.append(train_dice)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            model_saved_msg = "Model saved based on validation loss!"
            print(model_saved_msg)
            log_to_file(log_path, model_saved_msg)

        epochs_range = range(1, epoch + 2)
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks([epoch for epoch in epochs_range if epoch % 10 == 1 or epoch == max(epochs_range)])

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_dices, label='Train Dice')
        plt.plot(epochs_range, val_dices, label='Validation Dice')
        plt.legend()
        plt.title('Dice Score')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.xticks([epoch for epoch in epochs_range if epoch % 10 == 1 or epoch == max(epochs_range)])

        plt.savefig(os.path.join(training_path, 'progress.png'))
        plt.close()

if __name__ == "__main__":
    main()