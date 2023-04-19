


from breaststudies.models import BasicModel
import monai.networks.nets as nets


class UNet(BasicModel):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        roi_size,
        spatial_dims=3,
        **kwargs
    ):
        super().__init__(in_ch, out_ch, roi_size, **kwargs)
        self.model = nets.BasicUNet(spatial_dims=spatial_dims, in_channels=in_ch, out_channels=out_ch )
        self.roi_size = roi_size
   
    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor, []

   

