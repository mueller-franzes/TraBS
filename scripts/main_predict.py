from pathlib import Path
from datetime import datetime
from shutil import copyfile
import logging

import numpy as np 
import torch 
import torch.nn.functional as F
import SimpleITK as sitk 
import torchio as tio 

from breaststudies.data import BreastDatasetCreator
from breaststudies.models import UNet, nnUNet, SwinUNETR
from breaststudies.augmentation import Resample2, ZNormalization,  AddBlankChannel
from breaststudies.postprocessing import keep_connected, close_holes, keep_inside
from breaststudies.utils.prediction import series_pred


#--------------------------------------------------------------------------------------------
# This script loads the model of each fold and predicts the segmentation masks as an ensemble  
#--------------------------------------------------------------------------------------------



for dataset_name in ['duke',]: # 'uka','duke', 'breast-diagnosis' 
    # dataset_name = 'duke'     # 'uka', 'duke', 'breast-diagnosis'
    cohort = 'subset'        # 'subset', 'entire'
    
    for target_name in ['tissue']:   # 'tissue' , 'breast' 
        model_version = 0
        min_volume = 10**3        # Minimum breast segmentation volume  in mm^3
        test_time_flipping = True 

        Model =  nnUNet  # nnUNet SwinUNETR
        path_run_dir = {'tissue': Path.cwd() / 'runs/2023_02_05_232342_nnUNet_tissue', 
                        'breast':  Path.cwd()/'runs/2023_02_12_154048_nnUNet_breast'} 


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reference_file = {'uka': 'Dyn_0.nii', 'duke':'T1.nii.gz', 'breast-diagnosis':'T2.nii.gz' } # needed to extract Spacing, Origin, Direction 
        breast_mask_file = 'mask_breast.nii.gz' # needed for crop and possible preprocessing 


        series_trans = tio.Compose([ 
            # ToOrientation('LPS'), 
            # Resample2((0.64, 0.64, 3)) 
        ])
        item_trans = tio.Compose([
            ZNormalization(percentiles=(0.5, 99.5), per_channel=True, masking_method=lambda x:x>0),
            AddBlankChannel(1 if dataset_name=='duke' else 0) if dataset_name != "uka" else tio.Lambda(lambda x:x)
        ]) 

        ds = BreastDatasetCreator(
            dataset_name, 
            cohort, 
            lateral='unilateral', 
            target=target_name, 
            target_files={'mask':breast_mask_file} if target_name == 'tissue' else {},  
            out_format='tio', 
            item_trans=item_trans, 
            series_trans=series_trans,
        )

        # Set output dir 
        path_root_out = ds.path_root 

        # ------------------ Create Logger ----------------
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO)
        path_log_dir = Path().cwd()/'results'/dataset_name/cohort/'log'/'predict'
        path_log_dir.mkdir(parents=True, exist_ok=True)
        logger.addHandler(logging.FileHandler(path_log_dir/'logging.log', 'w'))

        # --------------------- Load Model(s) --------------
        models = []
        for path_split in path_run_dir[target_name].iterdir():
            if not (path_split.is_dir() and path_split.name.startswith('split')):
                continue

            model = Model.load_best_checkpoint(path_split, version=model_version)
            model.to(device)
            model.eval()
            models.append(model)


        # --------------- Iterate Cases --------------------
        # Note: Combine all items (eg. slices, sides,...) of a series/case and combine items to one prediction mask before 
        # proceed with next case  
        series_pointers = ds.get_series_pointers()
        for n_series, (series_id, item_pointers) in enumerate(series_pointers.items()):
            
            # Read Meta
            case_dir = item_pointers[0][0]
            logger.info(f"Case {n_series+1}: {series_id}")


            try:
                # Iterate over all models 
                pred_models = []
                for model_i , model in enumerate(models):
                    logger.info(f"Model {model_i} predicting")
                    
                    # Predict all items (e.g. left and right side of breast) and return combined mask
                    pred_mask, _ = series_pred(item_pointers, ds.load_item, model, test_time_flipping, device)

                    # Add prediction to models
                    pred_models.append(pred_mask)


                # Apply majority voting between models 
                pred_models = torch.stack(pred_models).type(torch.uint8)
                pred_mask = pred_models.mode(dim=0).values

                # Torch to Numpy
                pred_mask = pred_mask[0,0].cpu().numpy().astype(np.uint8)

                # Get meta data 
                path_case = ds.path_root/case_dir
                ref_img = sitk.ReadImage(str(path_case/reference_file[dataset_name] ))

                # Apply Postprocessing
                if target_name == 'breast':
                    label_values = ds.get_labels(exc_label_names='Background')
                    label_fcts = ds.get_label_fcts(exc_label_names='Background')
                    d,h,w = pred_mask.shape
                    for side in [(slice(None), slice(None), slice(w//2)), (slice(None), slice(None), slice(w//2, None))]:
                        pred_mask[side] = keep_connected(pred_mask[side], voxel_size=ref_img.GetSpacing(), min_volume=min_volume, 
                                                    keep_only_largest=label_values, label_fcts=label_fcts)
                        pred_mask[side] = close_holes(pred_mask[side], label_fcts=label_fcts, label_values=label_values)
                elif target_name == 'tissue':
                    mask_breast = sitk.GetArrayFromImage(sitk.ReadImage(str(path_case/breast_mask_file )))
                    pred_mask = keep_inside(pred_mask, mask_breast) # FGT can't exist outside breast-mask 

                # Create Nifti 
                pred_mask_nii = sitk.GetImageFromArray(pred_mask)
                pred_mask_nii.CopyInformation(ref_img) # copies the Origin, Spacing, and Direction
            
                # Write file 
                path_out = path_root_out/case_dir
                path_out.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(pred_mask_nii, str(path_out/f'mask_{target_name}_nn.nii.gz'))
            
            except Exception as e:
                logger.warning(f"Error: {e}")
    



   

    
