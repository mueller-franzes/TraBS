

from breaststudies.models import BasicModel
import breaststudies.models.monai_mods as nets



class SwinUNETR(BasicModel):
    def __init__(self, 
        in_ch, 
        out_ch, 
        roi_size,
        spatial_dims = 3,                      
        patch_sizes =  (         (1,2,2), (1,2,2), 2,  2),           
        kernel_sizes = ((1,3,3), (1,3,3),  3,      3,  3),
        kernel_size_emb = 2,
        depths=(2,2,2),
        num_heads=(2, 4, 8),
        feature_sizes =(12, 24, 48, 96, 192),
        deep_supervision=1,
        use_spacing=False,
        **kwargs
    ):
        super().__init__(in_ch, out_ch, roi_size, **kwargs)
        self.model = nets.SwinUNETR(
            img_size=roi_size,
            in_channels=in_ch,
            out_channels=out_ch,
            patch_sizes = patch_sizes,
            kernel_sizes = kernel_sizes,
            kernel_size_emb=kernel_size_emb,
            depths=depths,
            num_heads=num_heads,
            feature_sizes=feature_sizes,
            spatial_dims=spatial_dims,
            deep_supervision=deep_supervision,
            use_spacing=use_spacing
        )
        

    def forward(self, x_in, **kwargs):
        prediction, pred_ver = self.model(x_in, **kwargs)
        return prediction, pred_ver



    

