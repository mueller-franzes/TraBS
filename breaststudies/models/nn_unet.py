from breaststudies.models import BasicModel
import monai.networks.nets as nets


class nnUNet(BasicModel):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        roi_size,
        spatial_dims=3,
        kernel_size=[[1,3,3], [1,3,3],   3,    3,3], 
        strides=    [ 1,      [1,2,2], [1,2,2],2,2],
        upsample_kernel_size=None,
        filters=None,
        deep_supervision=1,
        res_block=False,
        **kwargs
    ):
        super().__init__(in_ch, out_ch, roi_size, **kwargs)
        upsample_kernel_size = strides[1:] if upsample_kernel_size is None else upsample_kernel_size
        self.model = nets.DynUNet(
            in_channels=in_ch, 
            out_channels=out_ch, 
            spatial_dims=spatial_dims, 
            kernel_size=kernel_size, #[input, down1, down2, ..., bottleneck] 
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            deep_supervision=deep_supervision>0, # WARNING: Original nnUnet scales the target masks down to low res. predictions and not low res. predictions up to target mask  
            deep_supr_num=deep_supervision,
            res_block=res_block
        )

 
    def forward(self, x_in, **kwargs):
       # NOTE: During interference, predictions of lower layers aren't used even if supervision is enabled (MONAI follows nnUnet implementations )
       # https://github.com/MIC-DKFZ/nnUNet/blob/6844361bb1dd60efb5f35112e248cf377902cd53/nnunet/training/network_training/nnUNetTrainerV2.py#L182 
        if (self.model.deep_supervision) and self.training:
            prediction = self.model(x_in)
            pred_hor = prediction[:,0]
            # pred_ver = [prediction[:,i] for i in range(1, prediction.shape[1])]
            pred_ver = self.model.heads
        else:
            pred_hor = self.model(x_in)
            pred_ver = self.model.heads 

        return pred_hor, pred_ver

    

    


    