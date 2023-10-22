import os
import torch
import numpy as np
import nibabel as nib
from preprocess import Dataset2D, Dataset3D, NormalizeIntensity
from torch.utils.data import DataLoader
from MyUNet import DeepUNet2D, UNet3D
from tqdm import tqdm
from torchmetrics.classification import Dice
from preprocess import test_loader, test_image_paths, test_label_paths 
from collections import defaultdict
from monai.losses import DiceLoss

MODE = '2d'
DEVICE = torch.device('cuda')
BEST_MODEL_PATH = '/work/ovens_lab/thaonguyen/uncertainty/training_results/2023-10-20_19-06-36/best_model.pth'
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

    slices_counts = get_slices_count(dataset)
    reconstructed_volume = defaultdict(list)
    probability_volume = defaultdict(list) 
    
    with torch.no_grad():
        for i, (batch_images, batch_labels) in enumerate(tqdm(loader, desc="Inference", ncols=100)):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)
            #outputs = outputs[:,1].round()
            loss = dice_ce_loss(outputs, batch_labels)

            print(f"Shape of outputs: {outputs.shape}")

            dice_value = 1 - loss
            running_dice += dice_value.item()
            
            for j in range(batch_images.size(0)): 
                volume_idx, _ = dataset.index_map[i * loader.batch_size + j]
                # Extract the raw probability values
                slice_prob = outputs[j, 1].cpu().numpy()[:, :, np.newaxis]

                # Threshold the probabilities to get the binary mask
                slice_mask = (slice_prob > 0.5).astype(np.float32)

                
                reconstructed_volume[volume_idx].append(slice_mask)
                probability_volume[volume_idx].append(slice_prob)
                
                if len(reconstructed_volume[volume_idx]) == slices_counts[volume_idx]:
                    print(reconstructed_volume[volume_idx][0].shape)
                    volume_3d = np.stack(reconstructed_volume[volume_idx], axis=2)
                    pred_nii = nib.Nifti1Image(volume_3d, affine=np.eye(4))
                    output_filename = os.path.basename(original_image_paths[volume_idx]).replace('.nii.gz', '_pred.nii.gz')
                    nib.save(pred_nii, os.path.join(OUTPUT_DIR, output_filename))
                    
                    prob_volume_3d = np.stack(probability_volume[volume_idx], axis=2)
                    prob_nii = nib.Nifti1Image(prob_volume_3d, affine=np.eye(4))
                    output_filename_prob = os.path.basename(original_image_paths[volume_idx]).replace('.nii.gz', '.npz')
                    np.savez_compressed(os.path.join(OUTPUT_DIR, output_filename_prob), prob_volume_3d)
                    
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
