from pathlib import Path
from shutil import copyfile
import logging
import sys 

import numpy as np 
import torch 
import torchio as tio 
import SimpleITK as sitk 
from monai.metrics import compute_meandice

from breaststudies.augmentation.augmentations import Resample2, ZNormalization, ToOrientation, RandomDisableChannel, RescaleIntensity
from breaststudies.data import  BreastDataModuleLR, BreastDataModule
from breaststudies.models import UNet, nnUNet, SwinUNETR
from breaststudies.utils import one_hot
from breaststudies.postprocessing import keep_connected, close_holes, keep_inside
from breaststudies.utils.prediction import series_pred


#--------------------------------------------------------------------------------------------
# This script loads the model of each fold and predicts the segmentation masks in the corresponding test fold. 
#--------------------------------------------------------------------------------------------

dataset_name = 'uka'     # 'uka', 
cohort = 'subset'        # 'subset', 
target_name = 'tissue'   # 'tissue' , 'breast' 
model_version = 0
min_volume = 10**3      # Minimum breast segmentation volume  in mm^3
test_time_flipping = True 

Model = SwinUNETR # UNet, nnUNet, SwinUNETR
path_run_dir = {'tissue': Path.cwd() / 'runs/2023_04_05_192603_SwinUNETR', 
                'breast':  Path.cwd()/'runs/2023_02_12_154048_nnUNet_breast'}   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=3, suppress=True)
breast_mask_file = 'mask_breast.nii.gz' 

path_out_root = Path().cwd()/'results'/dataset_name/cohort/'predictions'/target_name/(path_run_dir[target_name].name)
path_out_root.mkdir(parents=True, exist_ok=True)



# ------------ Load DataModule ----------------
series_trans = tio.Compose([ # Every spatial transformation must be have a reverse function, otherwise predicted masks will not fit to original images  
    # tio.ToCanonical(), 
    # ToOrientation('LPS'),
    # Resample2((0.64, 0.64, 3)) 
    # RandomDisableChannel(channels=(0,), p=1)
])
# Make sure you overwrite, otherwise the train transformations will be loaded. 
item_trans = ZNormalization(percentiles=(0.5, 99.5), per_channel=True, masking_method=lambda x:x>0)
dm = BreastDataModuleLR.load(path_run_dir[target_name], out_format='tio', target=target_name, item_trans=item_trans, series_trans=series_trans)



#---------------------- Cross-validation --------------------
path_out_pred = path_out_root/'predictions'


for split in range(0, 5):
    # ---------------- Logger ------------------
    path_log_dir = path_out_root/'log' 
    path_log_dir.mkdir(parents=True, exist_ok=True)
    path_log_file = path_log_dir/f'logging_split_{split}.log'
    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(path_log_file, 'w')
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[s_handler, f_handler])

    # -------- Setup Dataset ------------
    dm.setup_split('test', split=split)
    ds = dm.ds_test

    # ------------- Load Model ----------
    # model = Model.load_best_checkpoint(path_run_dir[target_name]/f'split_{split}', version=model_version)
    model = Model.load_from_checkpoint(path_run_dir[target_name]/f'split_{split}/last.ckpt', version=model_version)
    model.to(device)
    model.eval()


    # --------------- Iterate over all Series within fold --------------------------
    series_pointers = ds.get_series_pointers()
    dices = [] 
    y_pred = { lab_name:[] for lab_name in ds.labels.keys()}
    y_true = { lab_name:[] for lab_name in ds.labels.keys()}
    delta_times = []
    for n_series, (series_id, item_pointers) in enumerate(series_pointers.items()):
        # Read Meta
        case_dir = series_id
        logger.info(f"Split {split} Case {n_series+1}: {case_dir}")

        # Create output dir 
        path_out_dir = path_out_pred/case_dir 
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Predict all items (e.g. left and right side of breast) and return combined mask
        pred_mask, item_pred_times = series_pred(item_pointers, ds.load_item, model, test_time_flipping, device)

        # Store computation time for series 
        logger.info(f"Series Computation took  {sum(item_pred_times):.3f}s")
        delta_times.append(item_pred_times)

        # Torch to Numpy 
        pred_mask = pred_mask[0,0].cpu().numpy().astype(np.uint8)
            
        # Get target 
        target_file_name = ds.default_target_files[target_name]['target']
        path_item = ds.path_root/item_pointers[0][0]
        target_nii = tio.LabelMap(path_item/target_file_name)
        # target_nii = tio.ToCanonical()(target_nii)
        target_nii = target_nii.as_sitk()
        target = torch.as_tensor(sitk.GetArrayFromImage(target_nii)[None,None]).long()

       
        # ----------- Apply Postprocessing --------------
        if target_name == 'breast':
            label_values = ds.get_labels(exc_label_names='Background')
            label_fcts = ds.get_label_fcts(exc_label_names='Background')
            d,h,w = pred_mask.shape
            for side in [(slice(None), slice(None), slice(w//2)), (slice(None), slice(None), slice(w//2, None))]:
                pred_mask[side] = keep_connected(pred_mask[side], voxel_size=target_nii.GetSpacing(), min_volume=min_volume, 
                                            keep_only_largest=label_values, label_fcts=label_fcts)
                pred_mask[side] = close_holes(pred_mask[side], label_fcts=label_fcts, label_values=label_values)
        # elif target_name == 'tissue':
        #     mask_breast = sitk.GetArrayFromImage(sitk.ReadImage(str(path_item/breast_mask_file )))
        #     pred_mask = keep_inside(pred_mask, mask_breast) # FGT can't exist outside breast-mask 


        # --------------------- Save prediction on disk ------------------------
        pred_mask_nii = sitk.GetImageFromArray(pred_mask)
        pred_mask_nii.CopyInformation(target_nii) # copies the Origin, Spacing, and Direction
        sitk.WriteImage(pred_mask_nii, str(path_out_dir/f'mask_{target_name}_nn.nii.gz'))




        # --------------------- (Optional) Performance Metrics ---------------------
        pred_mask = torch.as_tensor(pred_mask[None, None], dtype=torch.long)

        # Hot Fix 
        target[target==5] = 1
        target[target==6] = 3

        target_onehot = one_hot(target[:,0], num_classes=len(ds.labels))# .type(source.dtype)
        pred_onehot = one_hot(pred_mask[:,0], num_classes=len(ds.labels)) # [Batch, Classes, D, H, W]
     
        dice_score = compute_meandice(pred_onehot, target_onehot, ignore_empty=False)[0]
        dice_score = dice_score.cpu().numpy().flatten()
        dices.append(dice_score)
        logger.info(f"Dice {dice_score}")

        for label_name, label_val in ds.labels.items():
            y_pred[label_name].append(label_val in pred_mask)
            y_true[label_name].append(label_val in target)


        logger.info("")

    # -------------------- (Optional) Performance Evaluation ------------------------------
    delta_times = np.asarray(delta_times) #[items, rois]
    delta_times_items = np.sum(delta_times, 1)

    logger.info(f"Mean Computation took of Items {np.mean(delta_times_items):.3f} ± {np.std(delta_times_items):.3f} s")
    logger.info(f"Mean Computation took of Series {np.mean(delta_times):.3f} ± {np.std(delta_times):.3f} s")

    dices = np.asarray(dices)
    if len(dices) == 0:
        continue
    for label_name, label_val in ds.labels.items():
        dice = dices[:, label_val]
        logger.info("Dice {}: {:.3f} ± {:.3f}; {:.3f} [{:.3f}, {:.3f}] Min {:.3f}, Max {:.3f}".format(label_name, np.mean(dice),np.std(dice),  *np.percentile(dice, q=[50, 2.5, 97.5]), np.min(dice), np.max(dice)) )

    for label_name, label_val in ds.labels.items():
        if label_val == 0:
            continue
        y_pred_lab = np.asarray(y_pred[label_name])
        y_true_lab = np.asarray(y_true[label_name])
        tp = np.sum( (y_true_lab==1) & (y_pred_lab==1) )
        fp = np.sum( (y_true_lab==0) & (y_pred_lab==1) )
        fn = np.sum( (y_true_lab==1) & (y_pred_lab==0) )
        tn = np.sum( (y_true_lab==0) & (y_pred_lab==0) )
        conf_matrix = [ [tp, fp], [fn, tn]  ]
        logger.info(f"Label {label_name} {np.sum(y_true_lab)}")
        logger.info("Confusion matrix {}".format(conf_matrix))
        logger.info("Sensitivity {:.2f}".format(tp/(tp+fn+1e-9)))
        logger.info("1- Spec {:.2f}".format(1-tn/(tn+fp+1e-9)))
        