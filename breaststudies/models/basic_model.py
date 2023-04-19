
from pathlib import Path
import json
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_msssim import ssim
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import  compute_meandice 

from breaststudies.utils import one_hot, tensor_mask2image, tensor2image





class BasicModel(pl.LightningModule):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        roi_size,
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-2},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        loss=DiceCELoss,
        loss_kwargs={'include_background':False, 'softmax':True,  'to_onehot_y':True, 'batch':False, 'smooth_nr':1e-5, 'smooth_dr':1e-5},
        target_type='segmentation', # [segmentation, image, vector]
        sample_every_n_steps = 1000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_ch = in_ch 
        self.out_ch = out_ch
        self.roi_size = roi_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler 
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.sample_every_n_steps = sample_every_n_steps
        self.kwargs = kwargs

        self._step_train = -1
        self._step_val = -1
        self._step_test = -1


        self.loss_fct = loss(**loss_kwargs)
        self.target_type = target_type

    def forward(self, x_in, **kwargs):
        # raise NotImplementedError
        return x_in, [x_in] # Only dummy code - should return pred_horizontal, [pred_vertical]
    
    def predict(self, x_in, **kwargs):
        """Get final prediction (not vertical predictions, etc.) """
        return self(x_in, **kwargs)[0]
    
    def infer(self, x_in, **kwargs):
        # NOTE: If x_in has shape > patch_shape, input is separated into multiple overlapping patches which are fused 
        return sliding_window_inference(x_in, self.roi_size, 2, self.predict, 0.5, "gaussian", **kwargs)

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        source, target = batch['source'], batch['target']
        batch_size = source.shape[0]
        interpolation_mode = 'nearest-exact' # if self.target_type == 'segmentation' else 'area'
        
        # Run Model 
        pred, pred_vertical = self(source, spacing=batch.get('spacing', None))

        # Only relevant for image2image training: if model is float16, pred is also float16 but target (image) is float32
        if (target.dtype==torch.float32) and (pred.dtype == torch.float16):
            target = target.type(torch.float16)
        
        # ------------------------- Compute Loss ---------------------------
        logging_dict = {}
        logging_dict['loss'] = self.loss_fct(pred, target)

        for i, pred_i in enumerate(pred_vertical): 
            weight = 1 #torch.prod(torch.tensor(pred_i.shape))/torch.prod(torch.tensor(pred.shape))
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            logging_dict['loss'] += self.loss_fct(pred_i, target_i)*weight

        # # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            if self.target_type == "segmentation":
                target_onehot = one_hot(target[:,0], num_classes=self.out_ch)
                pred_soft = F.softmax(pred,  dim=1)
                pred_mask = torch.argmax(pred_soft, keepdim=True, dim=1).type(target.dtype)
                pred_onehot =  one_hot(pred_mask[:,0], num_classes=self.out_ch)

                dice = compute_meandice(pred_onehot, target_onehot, include_background=True, ignore_empty=False)[0]
                soft_dice = compute_meandice(pred_soft, target_onehot, include_background=True, ignore_empty=False)[0]
                logging_dict['dice'] = torch.mean(dice)
                logging_dict['softdice'] = torch.mean(dice)
                logging_dict['ce'] = F.cross_entropy(pred, target[:,0].long())

                for label_n in range(self.out_ch): 
                    hotlabel = torch.zeros(self.out_ch, device=pred.device)
                    hotlabel[label_n] = 1 
                    logging_dict['dice_label_'+str(label_n)] = dice[label_n]
                    logging_dict['softdice_label_'+str(label_n)] = soft_dice[label_n]
                    logging_dict['ce_label_'+str(label_n)] = F.cross_entropy(pred, target[:,0].long(), reduction='sum', weight=hotlabel)
                
                    if state == "val" and (label_n>1):
                        true_lab =  (label_n in target)
                        pred_lab = (label_n in pred_mask)
                        logging_dict['tp_'+str(label_n)] = float(true_lab and pred_lab)
                        logging_dict['tn_'+str(label_n)] = float((not true_lab) and (not pred_lab))
                        logging_dict['fp_'+str(label_n)] = float((not true_lab) and pred_lab)
                        logging_dict['fn_'+str(label_n)] = float(true_lab and (not pred_lab))
                        logging_dict['acc_'+str(label_n)] = (logging_dict['tp_'+str(label_n)]+logging_dict['tn_'+str(label_n)])/1


            
            elif self.target_type == "image":
                logging_dict['L1'] = F.l1_loss(pred, target)
                logging_dict['L2'] = F.mse_loss(pred, target)               
                logging_dict['SSIM'] = ssim((pred+1)/2, (target+1)/2, data_range=1, nonnegative_ssim=True)

            


            
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val.cpu() if hasattr(metric_val, 'cpu') else metric_val, batch_size=batch_size, on_step=True, on_epoch=True)   
    
        
            #------------------ Log Image -----------------------
            if (step % self.sample_every_n_steps == 0):
                log_step = step // self.sample_every_n_steps
                path_out = Path(self.logger.log_dir)/'images'/state
                path_out.mkdir(parents=True, exist_ok=True)

                if self.target_type == "segmentation":
                    images = tensor_mask2image(source, pred_onehot)
                    save_image(images, path_out/f'sample_{log_step}.png',  nrow=images.shape[0]//source.shape[1], normalize=True, scale_each=True)

                elif self.target_type == "image":
                    images = torch.cat([tensor2image(img)[:32] for img in (source, target, pred)]) 
                    save_image(images, path_out/f'sample_{log_step}.png',  nrow=images.shape[0]//3, normalize=True, scale_each=True)
        
  
           
        return logging_dict['loss'] 

    def training_step(self, batch: dict, batch_idx: int):
        self._step_train += 1 
        return self._step(batch, batch_idx, "train", self._step_train)
        
    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val)

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)
    
    # def training_epoch_end(self, outputs):
    #     return 

    # def validation_epoch_end(self, outputs):
    #     return

    # def test_epoch_end(self, outputs):
    #     return

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def get_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        path_version = 'lightning_logs/version_'+str(version)
        with open(Path(path_checkpoint_dir) / path_version/ 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        path_best_checkpoint = cls.get_best_checkpoint(path_checkpoint_dir, version)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)


    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self.get_best_checkpoint(checkpoint_path, **kwargs)  
 
        with pl_legacy_patch():
            if map_location is not None:
                checkpoint = pl_load(checkpoint_path, map_location=map_location)
            else:
                checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        return self.load_weights(checkpoint["state_dict"], **kwargs)
    
    def load_weights(self, pretrained_weights, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=True)
        return self 



    