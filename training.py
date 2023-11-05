import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from mcdropout import MCDropout3D
from preprocess import Dataset2D, Dataset3D
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from MyUNet import UNet3D, DeepUNet2D
from monai.losses import DiceLoss
from torchmetrics.classification import Dice
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocess import train_loader, valid_loader, train_image_paths, train_label_paths, valid_image_paths, valid_label_paths
from collections import defaultdict
from torch.cuda import memory_summary

# Constants
NUM_EPOCHS = 500
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

dice_ce_loss = DiceLoss(to_onehot_y=True)

# ================= Training Utilities =================

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for batch_images, batch_labels in tqdm(loader, desc="Training", ncols=100):
        # print(memory_summary(device=DEVICE, abbreviated=False))  # Before operation
        
        batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(batch_images).to(DEVICE)
        loss = dice_ce_loss(outputs, batch_labels)
        
        loss.backward()
        optimizer.step()

        # print(memory_summary(device=DEVICE, abbreviated=False))  # After operation

        # threshold_outputs = (outputs >= 0.5).float()
        dice_value = 1 - loss
        running_dice += dice_value.item()

        running_loss += loss.item()

    return running_loss / len(loader), running_dice / len(loader)

def save_validation_predictions(model, loader, original_image_paths, val_predictions_dir, val_probabilities_dir, epoch, dataset):
    model.eval()
    reconstructed_volume = defaultdict(list)
    probability_volume = defaultdict(list)
    
    if not os.path.exists(val_predictions_dir):
        os.makedirs(val_predictions_dir)

    if not os.path.exists(val_probabilities_dir):
        os.makedirs(val_probabilities_dir)

    # Calculate the expected number of slices per volume
    expected_slices_per_volume = defaultdict(int)
    for path in dataset.image_paths:
        volume_idx = os.path.basename(path).split('_')[2]
        expected_slices_per_volume[volume_idx] += 1
    print("expected_slices_per_volume: ", expected_slices_per_volume)

    with torch.no_grad():
        for batch_idx, (batch_images, _) in enumerate(tqdm(loader, desc="Saving Validation Predictions", ncols=100)):
            batch_images = batch_images.to(DEVICE)
            outputs = model(batch_images)
            
            for j, image in enumerate(batch_images):
                image_path_idx = batch_idx * loader.batch_size + j
                #print(f"Processing slice {j} in batch {batch_idx}, image path index: {image_path_idx}")  # This should be inside the loop
            
                #print("len(dataset.image_paths): ", len(dataset.image_paths))
                if image_path_idx >= len(dataset.image_paths):
                    break 
                
                filename = os.path.basename(dataset.image_paths[image_path_idx])
                volume_idx = filename.split('_')[2]

                # Extract the raw probability values
                slice_prob = outputs[j, 1].cpu().numpy()[:, :, np.newaxis]

                # Debug: Save the probability map to a text file for inspection
                prob_map_filename = f"prob_map_{volume_idx}_slice{j}_epoch{epoch+1}.txt"
                prob_map_filepath = os.path.join(val_probabilities_dir, prob_map_filename)
                np.savetxt(prob_map_filepath, slice_prob[:,:,0], fmt='%f')

                # Threshold the probabilities to get the binary mask
                slice_mask = (slice_prob > 0.5).astype(np.float32)

                reconstructed_volume[volume_idx].append(slice_mask)
                probability_volume[volume_idx].append(slice_prob)
                
                print(f"Volume {volume_idx}: Processed {len(reconstructed_volume[volume_idx])} slices, Expected {expected_slices_per_volume[volume_idx]} slices.")

                # Check if all slices for the current volume have been processed
                if len(reconstructed_volume[volume_idx]) == expected_slices_per_volume[volume_idx]:
                    volume_3d = np.stack(reconstructed_volume[volume_idx], axis=2)
                    pred_nii = nib.Nifti1Image(volume_3d, affine=np.eye(4))

                    output_filename_pred = f"validation_pred_{volume_idx}_epoch{epoch+1}.nii.gz"
                    output_filepath = os.path.join(val_predictions_dir, output_filename_pred)
                    nib.save(pred_nii, output_filepath)

                    print(f"Volume {volume_idx} saved with {len(reconstructed_volume[volume_idx])} slices.")

                    del reconstructed_volume[volume_idx]
                    del probability_volume[volume_idx]

                else:
                    print(f"Volume {volume_idx} not yet complete. {len(reconstructed_volume[volume_idx])}/{expected_slices_per_volume[volume_idx]} slices processed.")

def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for batch_images, batch_labels in tqdm(loader, desc="Validation", ncols=100):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)
            loss = dice_ce_loss(outputs, batch_labels)

            dice_value = 1 - loss
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

    base_path = '/work/ovens_lab/thaonguyen/uncertainty'
    val_predictions_dir = os.path.join(base_path, 'val_predictions(with prob)')
    val_probabilities_dir = os.path.join(base_path, 'val_prob')

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

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        epoch_msg = f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
        print(epoch_msg)
        log_to_file(log_path, epoch_msg)

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_dice = evaluate(model, valid_loader)

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved based on validation dice: {best_dice:.4f} at epoch {epoch+1}")
            log_to_file(log_path, f"Model saved based on validation dice: {best_dice:.4f} at epoch {epoch+1}")

            # Clear previous predictions
            if os.path.exists(val_predictions_dir):
                for f in os.listdir(val_predictions_dir):
                    os.remove(os.path.join(val_predictions_dir, f))

            if os.path.exists(val_probabilities_dir):
                for f in os.listdir(val_probabilities_dir):
                    os.remove(os.path.join(val_probabilities_dir, f))

            # Save new best validation predictions
            save_validation_predictions(model, valid_loader, valid_image_paths, val_predictions_dir, val_probabilities_dir, epoch, valid_loader.dataset)

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
