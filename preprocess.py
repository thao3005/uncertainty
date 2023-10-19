import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class Dataset3D(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        img = nib.load(img_path)
        lbl = nib.load(label_path)
        img_data = img.get_fdata()
        lbl_data = lbl.get_fdata()
        img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_data).float().unsqueeze(0)
        lbl_tensor = lbl_tensor.to(dtype=torch.int32)
        
        if self.transform:
            img_tensor, lbl_tensor = self.transform(img_tensor, lbl_tensor)
        
        return img_tensor, lbl_tensor

class Dataset2D(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        assert len(image_paths) == len(label_paths), "Mismatched lengths of image_paths and label_paths"
        
        for path in image_paths:
            assert os.path.exists(path), f"Image file {path} does not exist"
        for path in label_paths:
            assert os.path.exists(path), f"Label file {path} does not exist"

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

        self.index_map = []
        self.img_slices_cache = []
        self.lbl_slices_cache = []

        for idx, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            img_3d = nib.load(img_path, mmap=False).get_fdata()
            lbl_3d = nib.load(lbl_path, mmap=False).get_fdata()

            depth = img_3d.shape[2]
            for slice_idx in range(depth):
                self.index_map.append((idx, slice_idx))
                img_slice = img_3d[:, :, slice_idx]
                lbl_slice = lbl_3d[:, :, slice_idx]
                self.img_slices_cache.append(img_slice)
                self.lbl_slices_cache.append(lbl_slice)

            del img_3d
            del lbl_3d

    def __len__(self):
        return len(self.img_slices_cache)

    def __getitem__(self, idx):
        img_slice = self.img_slices_cache[idx]
        lbl_slice = self.lbl_slices_cache[idx]

        img_tensor = torch.tensor(img_slice).float().unsqueeze(0)
        lbl_tensor = torch.tensor(lbl_slice).float().unsqueeze(0)
        lbl_tensor = lbl_tensor.to(dtype=torch.int32)

        if self.transform:
            img_tensor, lbl_tensor = self.transform(img_tensor, lbl_tensor)

        return img_tensor, lbl_tensor
        
def NormalizeIntensity():
    def __call__(self, image, label):
        return image, label
        # return (image - image.min()) / (image.max() - image.min()), label

def gather_paths(data_folder, subfolder_name):
    image_folder = os.path.join(data_folder, subfolder_name, 'images')
    label_folder = os.path.join(data_folder, subfolder_name, 'labels')
    image_paths = sorted([os.path.join(image_folder, p) for p in os.listdir(image_folder) if p.endswith('.nii.gz')])
    label_paths = sorted([os.path.join(label_folder, p) for p in os.listdir(label_folder) if p.endswith('.nii.gz')])
    return image_paths, label_paths

def set_up(data_folder, MODE):
    TRAINING_SUBFOLDER = 'training'
    VALIDATION_SUBFOLDER = 'validation'
    TESTING_SUBFOLDER = 'testing'
    train_image_paths, train_label_paths = gather_paths(data_folder, TRAINING_SUBFOLDER)
    valid_image_paths, valid_label_paths = gather_paths(data_folder, VALIDATION_SUBFOLDER)
    test_image_paths, test_label_paths = gather_paths(data_folder, TESTING_SUBFOLDER)

    if MODE == '2d':
        train_dataset = Dataset2D(train_image_paths, train_label_paths, transform=NormalizeIntensity())
        valid_dataset = Dataset2D(valid_image_paths, valid_label_paths, transform=NormalizeIntensity())
        test_dataset = Dataset2D(test_image_paths, test_label_paths, transform=NormalizeIntensity())
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)

    elif MODE == '3d':
        train_dataset = Dataset3D(train_image_paths, train_label_paths, transform=NormalizeIntensity())
        valid_dataset = Dataset3D(valid_image_paths, valid_label_paths, transform=NormalizeIntensity())
        test_dataset = Dataset3D(test_image_paths, test_label_paths, transform=NormalizeIntensity())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader, train_image_paths, train_label_paths, valid_image_paths, valid_label_paths, test_image_paths, test_label_paths

# DATASET_FOLDER = "/work/ovens_lab/thaonguyen/uncertainty/dataset"
DATASET_FOLDER = "/home/seyedsina.ziaee/src/uncertainty/dataset"
MODE = "2d" 
train_loader, valid_loader, test_loader, train_image_paths, train_label_paths, valid_image_paths, valid_label_paths, test_image_paths, test_label_paths = set_up(DATASET_FOLDER, MODE)
