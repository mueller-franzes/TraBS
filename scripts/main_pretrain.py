
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import torchio as tio 


from breaststudies.data import BreastDataModule, BreastDataModuleLR, BreastDataModule2D, BreastUKADataset
from breaststudies.models import UNet, nnUNet, SwinUNETR
from breaststudies.augmentation import ZNormalization, RandomCropOrPad, RandomCutOut, RandomDisableChannel


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)

    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None
    torch.set_float32_matmul_precision('high')

    # ------------- Settings ---------------------
    target = 'tissue' # 'breast' or 'tissue' 
    batch_size = 1 
    roi_sizes = {
        'tissue': (32, 256, 256),
        'breast': (32, 512, 256),
    }
    roi_size = roi_sizes[target]
    # NOTE: Source files include target as source files are loaded as ScalarImage and target files as LabelImage
    source_files={
        'breast': {'source':['Dyn_0.nii', 'T2_resampled.nii', ], 'target':['Dyn_0.nii', 'T2_resampled.nii',]},
        'tissue': {'source':['Dyn_0.nii','T2_resampled.nii', 'Sub.nii' ], 'target':['Dyn_0.nii',  'T2_resampled.nii', 'Sub.nii' ]} 
    }


    # ---------------------------------- Preprocessing ----------------------------------
    series_trans = tio.Compose([ 
        # tio.ToCanonical(),
        # Resample2((0.64, 0.64, 3)), # exact (0.64453125, 0.64453125, 3)
        # RandomResample(),
        # SelectRandomChannel()
    ])

    # --------------------------------- Augmentation ---------------------------------------
    p=0.5
    item_trans = tio.Compose([
        ZNormalization(percentiles=(0.5, 99.5), per_channel=True, masking_method=lambda x:x>0),
        tio.RandomFlip(axes=(0,1,2), flip_probability=0.5, p=1),
        RandomCutOut((64, 64, 4), (32, 32, 2),  (4, 4, 1), patch_per='image', include=['source']),
        tio.RandomNoise(mean=0, std=(0, 0.75), p=p, include=['source']), 
        RandomCropOrPad((256, 256, 16)),
        RandomDisableChannel((0,1), disable_per='subject', p=p, include=['source', 'target'])
    ])


    # ----------------------- Load Data ----------------------------------
    dm = BreastDataModuleLR(
        # path_root = Path('/home/gustav/Documents/datasets/BreastDataset/Gustav'),
        path_root = Path('/mnt/hdd/datasets/breast/UKA/UKA_2021_05_25'),
        batch_size=batch_size, 
        target=target,  
        series_trans=series_trans,
        item_trans=item_trans,
        source_files=source_files[target], 
        target_files={'mask':'mask_breast_nn.nii.gz'}, 
        target_shape=roi_size[::-1], # The bounding box of the breast mask is enlarged (if smaller) to target_shape to prevent padding with zeros. Only used for target=='tissue'. 
    ) 

    # Load fixed,balanced split  
    # dm._item_pointers_split = dm.load_split(Path.cwd() / 'runs/splits/BreastDatasetLR'/dm.ds_kwargs['target'], 
    #                                         split_file=f'data_split_{dm.Dataset.__name__}.yaml' ) 
    

    dm.setup('fit') # Run GroupKFold if item_pointers aren't initialized yet 
    dm.save(path_run_dir) # Save setup configs 


    #---------------------- Cross-Fold --------------------
    for split in range(0, 5):# dm.n_splits
        path_split_dir = path_run_dir/('split_'+str(split))
        path_split_dir.mkdir(parents=True, exist_ok=True)
        dm.setup_split('fit', split=split) # Create train/val datasets for specific split 

        # --------------------------- Initialize Model ----------------------
        in_ch = len(dm.ds_train.kwargs['source_files']['source'])
        out_ch = in_ch 
        

        # -------- Choose model --------
        # model = BasicUNet(in_ch=in_ch, out_ch=out_ch, roi_size=roi_size, target_type='image')
        # model = nnUNet(in_ch=in_ch, out_ch=out_ch, roi_size=roi_size, target_type='image')
        model = SwinUNETR(in_ch=in_ch, out_ch=out_ch, roi_size=roi_size, target_type='image',
            use_spacing = False, # Use spacing as an additional input information  
        )
        model.loss_fct= torch.nn.MSELoss()




        # -------------- Training Initialization ---------------
        to_monitor = "train/loss" # WARNING: If log() is not called this parameter is ignored! 
        min_max = "min"

        early_stopping = EarlyStopping(
            monitor=to_monitor,
            min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
            patience=30, # number of checks with no improvement
            mode=min_max
        )
        checkpointing = ModelCheckpoint(
            dirpath=str(path_split_dir), # dirpath
            monitor=to_monitor,
            every_n_train_steps=50,
            save_last=True,
            save_top_k=1,
            mode=min_max,
        )
        trainer = Trainer(
            gpus=gpus,
            precision=16,
            gradient_clip_val=0.5,
            default_root_dir=str(path_split_dir),
            callbacks=[checkpointing, early_stopping],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=50, # 50
            auto_lr_find=False,
            # limit_train_batches=1.0,
            # limit_val_batches=1.0, # 0 = disable validation 
            min_epochs=100,
            max_epochs=1001,
            num_sanity_val_steps=2,
        )
        
        # ---------------- Execute Training ----------------
        trainer.fit(model, datamodule=dm)

        # ------------- Save path to best model -------------
        model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)

        del trainer
