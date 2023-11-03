import os
import torch
import numpy as np
import nibabel as nib
from preprocess import Dataset2D, Dataset3D
from torch.utils.data import DataLoader
from MyUNet import DeepUNet2D, UNet3D
from tqdm import tqdm
from torchmetrics.classification import Dice
from preprocess import test_loader, test_image_paths, test_label_paths 
from collections import defaultdict
from monai.losses import DiceLoss

MODE = '2d'
DEVICE = torch.device('cuda')
BEST_MODEL_PATH = '/work/ovens_lab/thaonguyen/uncertainty/training_results/2023-10-30_09-11-20/best_model.pth'
OUTPUT_DIR = '/work/ovens_lab/thaonguyen/uncertainty/predictions'

def dice_coefficient(target, preds):
    temp = torch.zeros_like(preds[:, 1])
    temp[temp > 0.5] = 1
    preds = temp.unsqueeze(1)
    intersection = (preds * target).sum().float()
    set_sum = preds.sum() + target.sum()
    
    dice = (2 * intersection + 1e-8) / (set_sum + 1e-8) 
    
    return dice.item() 

dice_ce_loss = DiceLoss(to_onehot_y=True)

def load_model(mode, path):
    if mode == '2d':
        model = DeepUNet2D().to(DEVICE)
    elif mode == '3d':
        model = UNet3D().to(DEVICE)
    
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def infer_and_save(model, loader, original_image_paths, dataset):
    running_dice = 0.0

    reconstructed_volume = defaultdict(list)
    probability_volume = defaultdict(list) 
    
    volume_idx_to_list_idx = {os.path.basename(path).split('_')[2]: i for i, path in enumerate(original_image_paths)}

    expected_slices_per_volume = defaultdict(int)
    for path in dataset.image_paths:
        volume_idx = os.path.basename(path).split('_')[2]
        expected_slices_per_volume[volume_idx] += 1

    with torch.no_grad():
        for i, (batch_images, batch_labels) in enumerate(tqdm(loader, desc="Inference", ncols=100)):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)
            loss = dice_ce_loss(outputs, batch_labels)

            dice_value = 1 - loss
            running_dice += dice_value.item()
            
            for j in range(batch_images.size(0)): 
                # Extract volume index from the filename
                filename = os.path.basename(dataset.image_paths[i * loader.batch_size + j])
                volume_idx = filename.split('_')[2]  

                # Extract the raw probability values
                slice_prob = outputs[j, 1].cpu().numpy()[:, :, np.newaxis]

                # Threshold the probabilities to get the binary mask
                slice_mask = (slice_prob > 0.5).astype(np.float32)

                reconstructed_volume[volume_idx].append(slice_mask)
                probability_volume[volume_idx].append(slice_prob)
                
                # Debugging print statement
                # print(f"Processed slice {j+1} for volume {volume_idx}. Total slices so far: {len(reconstructed_volume[volume_idx])}")

                if len(reconstructed_volume[volume_idx]) == expected_slices_per_volume[volume_idx]:
                    volume_3d = np.stack(reconstructed_volume[volume_idx], axis=2)
                    pred_nii = nib.Nifti1Image(volume_3d, affine=np.eye(4))

                    base_filename = os.path.basename(original_image_paths[volume_idx_to_list_idx[volume_idx]])
                    base_filename_parts = base_filename.split('_')
                    base_filename_without_slice = '_'.join(base_filename_parts[:3])

                    # Save the prediction
                    output_filename_pred = f"{base_filename_without_slice}_pred.nii.gz"
                    nib.save(pred_nii, os.path.join(OUTPUT_DIR, output_filename_pred))
                    print(f"Saved prediction for volume {volume_idx}")

                    # Save the probability map as .nii.gz
                    prob_volume_3d = np.stack(probability_volume[volume_idx], axis=2)
                    prob_nii = nib.Nifti1Image(prob_volume_3d, affine=np.eye(4))
                    output_filename_prob_nii = f"{base_filename_without_slice}_prob.nii.gz"
                    nib.save(prob_nii, os.path.join(OUTPUT_DIR, output_filename_prob_nii))
                    print(f"Saved probability map as NIfTI for volume {volume_idx}")

                    # Save the probability map as .npz
                    output_filename_prob_npz = f"{base_filename_without_slice}_prob.npz"
                    np.savez_compressed(os.path.join(OUTPUT_DIR, output_filename_prob_npz), prob_volume_3d)
                    print(f"Saved probability map as NPZ for volume {volume_idx}")

                    del reconstructed_volume[volume_idx]
                    del probability_volume[volume_idx]
            
            del batch_images, batch_labels

    return running_dice / len(loader)

def get_slices_count(dataset):
    """Returns a list of slice counts for each 3D image in the dataset."""
    slice_counts = [0] * len(dataset.image_paths)
    for idx, slice_idx in dataset.index_map:
        slice_counts[idx] += 1
    return slice_counts

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    model = load_model(MODE, BEST_MODEL_PATH)
    avg_dice_score = infer_and_save(model, test_loader, test_image_paths, test_loader.dataset)
    
    print(f"Average Dice Score for Inference: {avg_dice_score:.4f}")

if __name__ == "__main__":
    main()
