import logging
from pathlib import Path 
import numpy as np 
import pandas as pd
import monai.metrics as mm 
import torchio as tio  

from breaststudies.utils import one_hot
from breaststudies.metrics import compute_surface_distances, compute_average_surface_distance
from breaststudies.data import BreastDatasetCreator, BreastUKADataset


dataset_name = 'duke'     # 'uka', 'duke', 'breast-diagnosis'
model = '2023_02_06_191619_nnUNet_breast'
cohort = 'subset'        # 'subset'
lateral='unilateral'      # 'bilateral' or 'unilateral'
target_name = 'tissue'   # 'tissue' , 'breast' 
use_2d = False 

path_root = Path().cwd()/'results'/dataset_name/cohort/'predictions'/target_name/model
path_predictions = path_root/'predictions'

path_out = path_root/'eval'
path_out.mkdir(parents=True, exist_ok=True)
path_log_dir = path_root/'log'
path_log_dir.mkdir(parents=True, exist_ok=True)




ds = BreastDatasetCreator(
    dataset_name, 
    cohort, 
    lateral=lateral,
    use_2d=use_2d, 
    out_format='torch', 
    source_files={}, 
    target=target_name, 
    item_trans=None, 
    series_trans=None,
    manipulate_label_func=BreastUKADataset.manipulate_label_func
)

# ds.labels.pop("DCIS")
eval_labels = ds.labels #{'FGT':1} 



# ------------ Logging --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_log_dir/'log_seg_quality.log', 'w'))


# Desired Metrics / Labels present 
results = {'UID':[]} 
results.update({f'{label}_Dice':[] for label in eval_labels.keys() })
results.update({f'{label}_ASSD':[] for label in eval_labels.keys() })
results.update({f'{label}_{rater}':[] for label in eval_labels.keys() for rater in ['NN', 'GT']})


for case_n, path_case_file in enumerate(path_predictions.rglob(f'mask_{target_name}_nn.nii.gz')):
    path_case_dir = path_case_file.relative_to(path_predictions).parent
    item_pointer = ((Path(path_case_dir)),)

    # Load Ground Truth and add Prediction to series  
    series = ds.load_series(item_pointer, ds.path_root, **ds.kwargs)
    series.add_image(tio.LabelMap(path_case_file), 'pred')

    # Split GT and Pred into desired format (eg. left/right or slices)
    series_items = ds.series2items(item_pointer[:max(1, len(item_pointer)-1)], series, **ds.kwargs)
    for item in series_items.values():
        # -------- Get data -------
        item = ds.item2out(item, **ds.kwargs)
        uid = item['uid'] 
        target = item['target']
        pred = item['pred']
        spacing = item['spacing'] # [x,y,z]
        logger.info(f"Case {case_n} UID: {uid}" )
                        
        # -------- Make one-hot --------
        target_onehot = one_hot(target, len(eval_labels))  # [B, C, D, H, W]
        pred_onehot = one_hot(pred, len(eval_labels)) # [B, C, D, H, W]

        # ------ Compute Metrics  -------
        results['UID'].append(uid)
        dice_score = mm.compute_meandice(pred_onehot, target_onehot, include_background=True, ignore_empty=False)[0]
        # assd_val = mm.compute_average_surface_distance(pred_onehot, target_onehot, include_background=True, symmetric=True) # Doesn't consider spacing 

        # Iterate over each label 
        for lab_name, lab_val in eval_labels.items():
            results[f'{lab_name}_GT'].append(lab_val in target)
            results[f'{lab_name}_NN'].append(lab_val in pred)
            results[f'{lab_name}_Dice'].append(dice_score[lab_val].item())
            try:
                surface_distances = compute_surface_distances(target_onehot[0,lab_val].numpy().astype(bool), pred_onehot[0,lab_val].numpy().astype(bool), spacing[::-1])
                results[f'{lab_name}_ASSD'].append(np.mean(compute_average_surface_distance(surface_distances)))
            except:
                results[f'{lab_name}_ASSD'].append(float('NaN')) # eg. if label is not present 
            
    
        logger.info("")




df = pd.DataFrame(results)
for label in eval_labels.keys():
    logger.info(f"{label} Dice (mean±std): {df[f'{label}_Dice'].mean():.3f} ± {df[f'{label}_Dice'].std():.3f}")
    logger.info(f"{label} Dice (min, max): {df[f'{label}_Dice'].min():.3f} to {df[f'{label}_Dice'].max():.3f} ")
df = df.set_index('UID', drop=True)
df.to_csv(path_out/f'segmentation_{lateral}_2D{use_2d}.csv')


