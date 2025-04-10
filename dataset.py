import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import concurrent.futures
from tqdm import tqdm

from utils import data_augmentation

def normalize(data):
    """Normalize data to range [0, 1]"""
    return data / 255.

def Im2Patch(img, win, stride=1):
    """Extract patches from an image with specified window size and stride"""
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def process_train_file(args):
    """Process a single training file and return generated patches
    
    Args:
        args: Tuple containing (file_path, scales, patch_size, stride, aug_times)
        
    Returns:
        List of tuples (data, name) where name is used for dataset creation
    """
    file_path, scales, patch_size, stride, aug_times = args
    results = []
    
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read {file_path}")
            return results
            
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(w*scales[k]), int(h*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                # Add base patch
                results.append((data, "base"))
                
                # Add augmented patches
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    results.append((data_aug, f"aug_{m+1}"))
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return results

def process_val_file(file_path):
    """Process a single validation file
    
    Args:
        file_path: Path to the validation image
        
    Returns:
        Processed image as numpy array, or None if error
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read {file_path}")
            return None
            
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        return img
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_data(data_path, patch_size, stride, aug_times=1, num_workers=4):
    """Prepare training and validation data in H5 files using multi-threading
    
    Args:
        data_path: Base path for training and validation data
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        aug_times: Number of augmentations per patch (default: 1)
        num_workers: Number of worker threads (default: 4)
    """
    # Training data preparation
    print('Preparing training data with multi-threading')
    scales = [1, 0.9, 0.8, 0.7]
    train_files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    train_files.sort()
    
    if not train_files:
        print(f"Warning: No training files found in {os.path.join(data_path, 'train')}")
        return
    
    # Prepare arguments for parallel processing
    train_args = [(f, scales, patch_size, stride, aug_times) for f in train_files]
    
    # Create H5 file for training data
    with h5py.File('train.h5', 'w') as h5f:
        train_num = 0
        
        # Process training files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_train_file, train_args),
                total=len(train_args),
                desc="Processing training files"
            ))
            
            # Create datasets from results
            for file_results in tqdm(results, desc="Writing training data to H5"):
                for data, suffix in file_results:
                    if suffix == "base":
                        dataset_name = str(train_num)
                    else:
                        dataset_name = f"{train_num}_{suffix}"
                    
                    h5f.create_dataset(dataset_name, data=data)
                    train_num += 1
    
    # Validation data preparation
    print('\nPreparing validation data with multi-threading')
    val_files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    val_files.sort()
    
    if not val_files:
        print(f"Warning: No validation files found in {os.path.join(data_path, 'Set12')}")
        return
    
    # Create H5 file for validation data
    with h5py.File('val.h5', 'w') as h5f:
        val_num = 0
        
        # Process validation files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(process_val_file, f): f for f in val_files}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                               total=len(val_files), 
                               desc="Processing validation files"):
                file_path = future_to_file[future]
                try:
                    img = future.result()
                    if img is not None:
                        h5f.create_dataset(str(val_num), data=img)
                        print(f"file: {os.path.basename(file_path)}")
                        val_num += 1
                except Exception as e:
                    print(f"Error finalizing {file_path}: {str(e)}")
    
    print(f'Training set, # samples {train_num}')
    print(f'Validation set, # samples {val_num}')

class Dataset(udata.Dataset):
    """PyTorch dataset for DnCNN training/validation"""
    
    def __init__(self, train=True):
        """Initialize the dataset
        
        Args:
            train: If True, use training data, otherwise use validation data
        """
        super(Dataset, self).__init__()
        self.train = train
        self.h5_file = 'train.h5' if self.train else 'val.h5'
        
        # Get keys but don't keep the file open
        with h5py.File(self.h5_file, 'r') as h5f:
            self.keys = list(h5f.keys())
            
        # Shuffle training data
        if self.train:
            random.shuffle(self.keys)
            
        # File handle to be opened in worker processes
        self.h5f = None
            
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        # Open file if not already open (lazy loading)
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_file, 'r')
            
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)
    
    def __del__(self):
        # Clean up resources
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None