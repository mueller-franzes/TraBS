import torch 
import torch.nn.functional as F
import torchio as tio 
import time 

import logging
logger = logging.getLogger(__name__)

def series_pred(item_pointers, load_item, model, test_time_flipping=False, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device 
    # Predict all masks within one series (e.g. left and right side of breast) 
    series_mask = [] 
    series_delta_times = []
    for item_pointer in item_pointers:

        # ------ Load Source File ---------
        item = load_item(item_pointer) 
        source_tio = item['source']  
        spacing = torch.tensor(item['spacing'], device=device)[None]
        source = source_tio.data.swapaxes(1,-1)
        source_batch = source[None] # add Batch-Dimension 
        source_batch = source_batch.to(device)


        # ---------- Run Model ------------
        with torch.no_grad(): 
            start_time = time.time()
            pred = model.infer(source_batch, spacing=spacing)
            roi_delta_time = time.time() - start_time
            logger.info(f"Computation took  {roi_delta_time:.3f}s")
            pred_prob = F.softmax(pred, dim=1)  # [Batch, Channels, D, H, W]
        
        # --------- Test Time Augmentation (Flip)
        if test_time_flipping: 
            pred_prob = pred_prob/8
            with torch.no_grad(): 
                for flip_axes in [(4,), (3,), (2,), (4, 3), (4, 2), (3,2), (4, 3, 2)]:
                    pred_prob += torch.flip(F.softmax(model.infer(torch.flip(source_batch, flip_axes), spacing=spacing), dim=1), flip_axes)/8

        # ---------- Soft 0-1 to hard prediction 0/1
        pred_mask =  torch.argmax(pred_prob, dim=1, keepdim=True).type(torch.long) # [Batch, 1, D, H, W]

        # ----------- Hot Fix --------------
        # pred_mask[pred_mask==2] = 3 # WARNING: Only valid for training purposes, uncomment otherwise

        # -------- Add prediction to TorchIO-subject --------
        pred_mask_tio = tio.LabelMap(tensor=pred_mask[0].swapaxes(1,-1).cpu(), affine=source_tio.affine)
        item.add_image(pred_mask_tio, 'prediction')
        if '_org_orientation_source' in item: # If ToOrientation was applied, add property for reverse process  
            item['_org_orientation_prediction'] = item['_org_orientation_source']

        # Reverse augmentation
        item_inv = item.apply_inverse_transform(warn=False) # -> get_inverse_transform()
        pred_mask = item_inv['prediction'].data.swapaxes(1,-1)[None]
            
        
        # Add item prediction to list for series prediction 
        series_mask.append(pred_mask)
        series_delta_times.append(roi_delta_time) 

    # Fusion all crops/items 
    pred_mask = torch.sum(torch.stack(series_mask), dim=0) # Note: summation is possible as e.g. contralateral side were padded with zeros 

    return pred_mask, series_delta_times