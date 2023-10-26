import os
import nibabel as nib

def gather_paths(data_folder, subfolder_name):
    image_folder = os.path.join(data_folder, subfolder_name, 'images')
    label_folder = os.path.join(data_folder, subfolder_name, 'labels')
    image_paths = sorted([os.path.join(image_folder, p) for p in os.listdir(image_folder) if p.endswith('.nii.gz')])
    label_paths = sorted([os.path.join(label_folder, p) for p in os.listdir(label_folder) if p.endswith('.nii.gz')])
    return image_paths, label_paths

def save_2d_slices(data_folder, subfolder_name, output_folder):
    image_paths, label_paths = gather_paths(data_folder, subfolder_name)
    
    img_output_folder = os.path.join(output_folder, subfolder_name, 'images')
    lbl_output_folder = os.path.join(output_folder, subfolder_name, 'labels')
    
    if not os.path.exists(img_output_folder):
        os.makedirs(img_output_folder)
    if not os.path.exists(lbl_output_folder):
        os.makedirs(lbl_output_folder)
    
    for img_path, lbl_path in zip(image_paths, label_paths):
        img_3d = nib.load(img_path)
        lbl_3d = nib.load(lbl_path)
        depth = img_3d.shape[2]
        
        img_index = os.path.basename(img_path).split('_')[1]
        
        for slice_idx in range(depth):
            # Save image slice
            img_slice = img_3d.dataobj[:, :, slice_idx]
            slice_filename = f"{subfolder_name}_image_{img_index}_{str(slice_idx).zfill(4)}.nii.gz"
            slice_path = os.path.join(img_output_folder, slice_filename)
            nib.save(nib.Nifti1Image(img_slice, img_3d.affine), slice_path)
            
            # Save label slice
            lbl_slice = lbl_3d.dataobj[:, :, slice_idx]
            slice_filename = f"{subfolder_name}_label_{img_index}_{str(slice_idx).zfill(4)}.nii.gz"
            slice_path = os.path.join(lbl_output_folder, slice_filename)
            nib.save(nib.Nifti1Image(lbl_slice, lbl_3d.affine), slice_path)

DATASET_FOLDER = "/work/ovens_lab/thaonguyen/uncertainty/dataset"
OUTPUT_FOLDER = "/work/ovens_lab/thaonguyen/uncertainty/2d_dataset"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for subfolder in ['training', 'validation', 'testing']:
    save_2d_slices(DATASET_FOLDER, subfolder, OUTPUT_FOLDER)
