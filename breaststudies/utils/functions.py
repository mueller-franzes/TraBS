import torch 
import torch.nn.functional as F 
import numpy as np 
from torchvision.utils import draw_segmentation_masks

def heaviside(input, threshold=0.5):
    """Heaviside function 
    
    Arguments:
        input {torch.Tensor} -- Input tensor
    
    Keyword Arguments:
        threshold {float} -- Input values greater or equal threshold are set to one (default: {0.5})
    
    Returns:
        torch.Tensor -- Binary tensor
    
    Warning:
        This function destroys the backprogation path ! 
    
    See also:
        PyTorch>1.7 https://pytorch.org/docs/1.7.0/generated/torch.heaviside.html#torch.heaviside
    
    """
    # Note: If 'requires_grad=True' , the backpropagation path will end in either 'upper' or 'lower' but not in 'input'  
    upper = torch.ones(1, device=input.device, requires_grad=False) 
    lower = torch.zeros(1, device=input.device , requires_grad=False)
    return torch.where(input>=threshold, upper, lower)

    

def one_hot(tensor,  num_classes=-1):
    """Wrapper for pytorch one-hot encoding
    
    Arguments:
        tensor {torch.Tensor} -- Tensor to be encoded of shape [Batch, (Depth), Height, Width]
    
    Keyword Arguments:
        num_classes {int} -- number of classes (default: {-1})
    
    Returns:
        torch.Tensor -- Tensor of shape [Batch, Classes, (Depth), Height, Width]
    
    Warning:
        This function destroys the backprogation path ! 
    """
    return F.one_hot(tensor.long(),num_classes).permute(0,tensor.ndim,*list(range(1,tensor.ndim)))



def minmax_norm(x, max=1, smooth_nr=1e-5, smooth_dr=1e-5):
    """Normalizes input to [0, max] for each beach and channel

    Args:
        x (torch.Tensor): Tensor to be normalized, Shape [Batch, Channel, *]

    Returns:
        torch.Tensor: Normalized tensor 
    """
    return torch.stack([ torch.stack([(ch-ch.min()+smooth_nr)/(ch.max()-ch.min()+smooth_dr)*max for ch in batch]) for batch in x])

def minmax_norm_slice(x, max=1, smooth_nr=1e-5, smooth_dr=1e-5):
    """Normalizes input to [0, max] for each beach, channel and slice 

    Args:
        x (torch.Tensor): Tensor to be normalized, Shape [Batch, Channel, *]

    Returns:
        torch.Tensor: Normalized tensor 
    """
    return torch.stack([ torch.stack([ torch.stack([(sl-sl.min()+smooth_nr)/(sl.max()-sl.min()+smooth_dr)*max for sl in ch]) for ch in batch]) for batch in x])


def tensor2image(tensor, batch=0):
    """Transform tensor into shape of multiple 2D RGB/gray images. 
        Keep 2D images as they are (gray or RGB).  
        For 3D images, pick 'batch' and use depth and interleaved channels as batch (multiple gray images). 

    Args:
        tensor (torch.Tensor): Image of shape [B, C, H, W] or [B, C, D, H, W]

    Returns:
        torch.Tensor: Image of shape [B, C, H, W] or [DxC,1, H, W]  (Compatible with torchvision.utils.save_image)
    """
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])


def tensor_mask2image(tensor, mask_hot, batch=0, alpha=0.25, colors=None, exclude_chs=[]):
    """Transform a tensor and a one-hot mask into multiple 2D RGB images.

    Args:
        tensor (torch.Tensor): Image tensor. Can be 3D volume of shape [B, C, D, W, H] or 2D of shape [B, C, H, W]
        mask_hot (torch.Tensor): One-Hot encoded mask of shape [B, Classes, D, W, H] or [B, Classes, H, W]
        batch (int, optional): Batch to use if input is 3D. Defaults to 0.
        alpha (float, optional): 1-Transparency. Defaults to 0.25.

    Returns:
        torch.Tensor: Tensor of 2D-RGB images with transparent mask on each. For 3D will be [CxD, 3, H, W] for 2D will be [B, 3, H, W] 
    """
    mask_hot = mask_hot.type(torch.bool).cpu() # To bool and cpu (see bug below)
    mask_hot = mask_hot if mask_hot.ndim<5 else torch.swapaxes(mask_hot[batch], 0, 1) # 3D [B, C, D, H, W] -> [C, D, H, W]. 2D [B, C, H, W] -> [B, C, H, W]
    image = minmax_norm(tensor, 255).type(torch.uint8).cpu() # To uint8 and cpu (see bug below)
    image = image[None] if image.ndim==4 else image[batch][:,:,None] # 3D [B, C, D, H, W] -> [C, D, 1, H, W]. 2D [B, C, H, W] -> [1, B, C, H, W] 
    image = torch.cat([image for _ in range(3)], dim=2) if image.shape[2]!=3 else image # Ensure RGB  [*, 3, H, W] 
    image = torch.stack([draw_segmentation_masks(i, m, alpha=alpha, colors=colors) if ch not in exclude_chs else i for ch, img_ch in enumerate(image)  for i,m in zip(img_ch, mask_hot) ]) # [B, 3, H, W]  # BUG Apparently only supports cpu()
    return image/255.0
