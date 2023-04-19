
from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import torchio as tio 

from breaststudies.data import BreastDataModule, BreastDataModuleLR, BreastDataModule2D, BreastUKADataset
from breaststudies.models import UNet, nnUNet, SwinUNETR
from breaststudies.augmentation import Resample2,RandomResample, ZNormalization, SpatialTransform2, RandomCropOrPad, Brightness, RandomDisableChannel, RescaleIntensity


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    torch.set_float32_matmul_precision('high')

    path_run_dir.mkdir(parents=True, exist_ok=True)
    

    # ------------- Settings ---------------------
    target = 'tissue' # 'breast' or 'tissue' 
    batch_size = 1 
    roi_sizes = {
        'tissue': (32, 256, 256),
        'breast': (32, 512, 256),
    }
    roi_size = roi_sizes[target]


    # ---------------------------------- Preprocessing ----------------------------------
    series_trans = tio.Compose([ 
        # tio.ToCanonical(),
        # Resample2((0.64, 0.64, 3)), # exact (0.64453125, 0.64453125, 3)
    ])
    item_trans = tio.Compose([ 
        ZNormalization(percentiles=(0.5, 99.5), per_channel=True, masking_method=lambda x:x>0),
        tio.CropOrPad(roi_size[::-1], padding_mode='minimum'),
    ])

    # --------------------------------- Augmentation ---------------------------------------
    p = 0.2 
    item_trans_train = tio.Compose([ 
        ZNormalization(percentiles=(0.5, 99.5), per_channel=True, masking_method=lambda x:x>0),
        # ZNormalization(per_channel=True, masking_method=lambda x:x>0),
        # RescaleIntensity((-1,1), percentiles=(0.5, 99.5), per_channel=True, masking_method=lambda x:x>0),
        # RescaleIntensity((-1,1), per_channel=True, masking_method=lambda x:x>0),
        tio.RandomGhosting(num_ghosts=(4, 10), axes=(0,), intensity=(0.25, 1), restore=0.02, p=p),
        tio.RandomFlip(axes=(0,1,2), flip_probability=0.5, p=1), #WARNING flip_probability = prob that specific axis is used , p = prob that trans is applied , 0 = left,right, 1 = buttom/top 
        SpatialTransform2(scales=(0.7,1.4, 0.7,1.4, 1.0,1.0), isotropic=(0,1), degrees=(0,0,0), translation=(0, 0, 0), default_pad_value = 0, image_interpolation='linear',  p=p),        
        tio.RandomNoise(mean=0, std=(0, 0.25), p=p), 
        tio.RandomBlur((1.0,1.0, 0), p=p), 
        tio.RandomBiasField(coefficients=0.1, p=p),  
        Brightness((0.75, 1.25), per_channel=True, p=p),
        tio.RandomGamma(log_gamma=(-0.4, 0.4), p=p), 
        RandomCropOrPad(roi_size[::-1], padding_mode='minimum'),
        RandomDisableChannel((0,1), p=p)
    ])


    # ------------ Optional ------------------
    # ds_kwargs = {
    #     'manipulate_label_func': BreastUKADataset.manipulate_label_func 
    # } if target == 'tissue' else {}


    # ----------------------- Load Data ----------------------------------
    dm = BreastDataModuleLR(
        path_root = Path('/home/gustav/Documents/datasets/BreastDataset'),
        batch_size=batch_size, 
        target=target,  
        series_trans=series_trans,
        item_trans=item_trans,
        params_ds_train={'item_trans':item_trans_train},
        # source_files={'source':['Dyn_0.nii', 'T2_resampled.nii', 'Sub.nii' ]}, # Overwrites default setting associated with 'target' setting  
        target_shape=roi_size[::-1], # The bounding box of the breast mask is enlarged (if smaller) to target_shape to prevent padding with zeros. Only used for target=='tissue'. 
        # num_workers=0,
        # **ds_kwargs        
    ) 

    # Load fixed,balanced split  
    # dm._item_pointers_split = dm.load_split(Path.cwd() / 'runs/splits/BreastDatasetLR'/dm.ds_kwargs['target'], 
    #                                         split_file=f'data_split_{dm.Dataset.__name__}.yaml' ) 
    

    dm.setup('fit') # Run GroupKFold if item_pointers aren't initialized yet 
    dm.save(path_run_dir) # Save setup configs 


    #---------------------- Cross-Fold --------------------
    for split in range(0, dm.n_splits):
        path_split_dir = path_run_dir/('split_'+str(split))
        path_split_dir.mkdir(parents=True, exist_ok=True)
        dm.setup_split('fit', split=split) # Create train/val datasets for specific split 

        # --------------------------- Initialize Model ----------------------
        in_ch = len(dm.ds_train.kwargs['source_files']['source'])
        out_ch = len(dm.ds_train.labels) # WARNING: manipulate_label_func might affect this 
        

        # -------- Choose model --------
        # model = BasicUNet(in_ch=in_ch, out_ch=out_ch, roi_size=roi_size)
        # model = nnUNet(in_ch=in_ch, out_ch=out_ch, roi_size=roi_size )
        model = SwinUNETR(in_ch=in_ch, out_ch=out_ch, roi_size=roi_size,
            use_spacing = False, # Use spacing as an additional input information  
            # use_checkpoint=True
        )


        # --------- Load pretraining or previously trained checkpoints ------------
        # model.load_pretrained(Path.cwd()/f'runs/2023_04_04_182914_SwinUNETR_pretrain/split_0/last.ckpt')
 
        

        # -------------- Training Initialization ---------------
        to_monitor = "val/loss" # WARNING: If log() is not called this parameter is ignored! 
        min_max = "min"

        early_stopping = EarlyStopping(
            monitor=to_monitor,
            min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
            patience=100, # number of checks with no improvement
            mode=min_max
        )
        checkpointing = ModelCheckpoint(
            dirpath=str(path_split_dir), # dirpath
            monitor=to_monitor,
            save_last=True,
            save_top_k=1,
            mode=min_max,
        )
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=None,
            precision=16,
            # amp_backend='apex',
            # amp_level='O2',
            gradient_clip_val=0.5,
            default_root_dir=str(path_split_dir),
            callbacks=[checkpointing,early_stopping ], # 
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=1, # 50
            auto_lr_find=False,
            # limit_train_batches=1.0,
            # limit_val_batches=0, # 0 = disable validation 
            # min_epochs=50,
            max_epochs=1001,
            num_sanity_val_steps=2,
        )
        
        # ---------------- Execute Training ----------------
        trainer.fit(model, datamodule=dm)

        # ------------- Save path to best model -------------
        model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)

        # del trainer
