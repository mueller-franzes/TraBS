
import scipy.ndimage as ndimage
import skimage.measure as measure 
import numpy as np 

def _keep_connected_binary(binary_mask, voxel_vol, min_volume=None, keep_only_largest=1):
    if keep_only_largest==0:
        return np.zeros(binary_mask.shape, dtype=binary_mask.dtype)

    mask_ind, num_features = measure.label(binary_mask, return_num=True) # ndimage.label(mask, structure=structure) # Each connected volume gets it's own label  
    label_sizes = {}
    for label in range(1, num_features+1):
        label_sizes[label] = np.sum(mask_ind==label)*voxel_vol
    original_num = len(label_sizes) 
    
    if min_volume is not None:
        label_sizes = {label:volume for label, volume in label_sizes.items() if volume > min_volume}
        print(label_sizes)
    
    label_sizes = dict(sorted(label_sizes.items(), key=lambda item: item[1], reverse=True)[0:keep_only_largest])

    if original_num == len(label_sizes):
        return binary_mask  # No need to change anything 

    if len(label_sizes)==0:
        return np.zeros(binary_mask.shape, dtype=binary_mask.dtype)
    else:
        valid_masks = np.stack([np.where(mask_ind==keep_label, 1, 0) for keep_label in label_sizes.keys()])
        return np.sum(valid_masks, axis=0, dtype=binary_mask.dtype)


def keep_connected(mask, voxel_size=(1,1,1), min_volume=None, keep_only_largest=1, label_fcts={'Foreground':lambda x:x>0}):
    voxel_size = np.asarray(voxel_size)
    voxel_vol = np.prod(voxel_size)

    new_mask = np.zeros(mask.shape, dtype=mask.dtype)
    for label_name, label_func in label_fcts.items():
        binary_mask = label_func(mask)
        _min_volume = min_volume[label_name] if isinstance(min_volume, dict) else min_volume
        _keep_only_largest = keep_only_largest[label_name] if isinstance(keep_only_largest, dict) else keep_only_largest
        binary_mask = _keep_connected_binary(binary_mask, voxel_vol, _min_volume, _keep_only_largest)
        new_mask = np.where(binary_mask, mask, new_mask)
    return new_mask 



def _close_holes_binary(binary_mask):
    # return ndimage.binary_closing(binary_mask).astype(binary_mask.dtype)  # WARNING: This also closes 'holes' that have an open side
    return ndimage.binary_fill_holes(binary_mask).astype(binary_mask.dtype) 
    
def close_holes(mask, label_fcts={'Foreground':lambda x:x>0}, label_values={'Foreground':1}):
    for label_name, label_func in label_fcts.items():
        binary_mask = label_func(mask)
        binary_mask = _close_holes_binary(binary_mask)
        mask = np.where(binary_mask, label_values[label_name], mask)
    return mask 


def keep_inside(mask, mask_border):
    mask[mask_border==0] = 0
    return mask 


