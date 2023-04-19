from typing import Iterable, Tuple, Union, List, Optional, Sequence, Dict
from numbers import Number
from pathlib import Path
import warnings

from tqdm import tqdm
import numpy as np 
import nibabel as nib 
import torch
import torchio as tio 
from torchio import Subject, RandomAffine, IntensityTransform, CropOrPad, Resample
from torchio import TYPE, INTENSITY, LABEL
from torchio.typing import TypeRangeFloat, TypeSextetFloat, TypeTripletFloat, TypeTripletInt, TypePath, TypeCallable
from torchio.transforms.transform import TypeMaskingMethod
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms.augmentation.spatial.random_affine import _parse_default_value
from torchio.utils import to_tuple
from torchio.transforms import SpatialTransform, Compose, Transform, HistogramStandardization

from breaststudies.augmentation.helper_functions import augment_linear_downsampling_scipy, augment_contrast, resample_patient
from breaststudies.utils import get_affine2


TypeOneToSixFloat = Union[TypeRangeFloat, TypeTripletFloat, TypeSextetFloat]


def parse_per_channel(per_channel, channels):
    if isinstance(per_channel, bool):
        if per_channel == True:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))] 
    else:
        return per_channel 

class ZNormalization(tio.ZNormalization):
    """Add option 'per_channel' to apply znorm for each channel independently and percentiles to clip values first"""
    def __init__(
        self,
        percentiles: TypeRangeFloat = (0, 100),
        per_channel=True,
        masking_method: TypeMaskingMethod = None,
        **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel


    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])

        image.set_data(torch.cat([
            self._znorm(image.data[chs,], mask[chs,], image_name, image.path)
            for chs in per_channel ])
        )
  

    def _znorm(self, image_data, mask, image_name, image_path):
        # NOTE: torch not reliable: "failed to apply transformation:  quantile() input tensor is too large"
        # cutoff2 = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles)/100.0)
        # torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)
        cutoff = np.percentile(image_data[mask], self.percentiles)
        np.clip(image_data, *cutoff, out=image_data.numpy())  # type: ignore[call-overload]

        standardized = self.znorm(image_data, mask)
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image_path})'
            )
            raise RuntimeError(message)
        return standardized



class RescaleIntensity(tio.RescaleIntensity):
    """Add option 'per_channel' to apply rescale for each channel independently"""
    def __init__(
        self,
        out_min_max: TypeRangeFloat = (0, 1),
        percentiles: TypeRangeFloat = (0, 100),
        masking_method: TypeMaskingMethod = None,
        in_min_max: Optional[Tuple[float, float]] = None,
        per_channel=True, # Bool or List of tuples containing channel indices that should be normalized together 
        **kwargs
    ):
        super().__init__(out_min_max, percentiles, masking_method, in_min_max, **kwargs)
        self.per_channel=per_channel

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        
        image.set_data(torch.cat([
            self.rescale(image.data[chs,], mask[chs,], image_name)
            for chs in per_channel ])
        )

        




class RandomCutOut(RandomTransform):
    """Cuts out rectangular boxes with random size and position and fill them with noise"""

    def __init__(self, patch_max, patch_min=None, patch_margin=None, patch_per='channel', fill_per_channel=True,  fill='random_image_values', fill_label=0, **kwargs):
        super().__init__(**kwargs)
        self.patch_max = np.asarray(patch_max) # Maximum patch sizes of shape [W, H, D]
        self.patch_min = np.asarray(patch_max if patch_min is None else patch_min) # Default to patch_max
        self.patch_margin = np.asarray(self.patch_min if patch_margin is None else patch_margin) # Default to patch_min 
        self.patch_per = patch_per # Equal random patches on 'subject', 'image', or 'channel' level 
        self.fill_per_channel = fill_per_channel # True - compute fill value on channel level, otherwise on image level 
        self.fill = fill
        self.fill_label = fill_label

    def apply_transform(self, subject: Subject) -> Subject:
        patches = None 
        if self.patch_per == 'subject':
            patches = self.get_patches(subject.spatial_shape)
        
        for image in subject.get_images(intensity_only=False, **self.add_include_exclude({})): 
            if self.patch_per == 'image':
                patches = self.get_patches(image.shape[1:])                   
            if image[TYPE] == INTENSITY:
                new_image = self.fill_rect(image.data, patches, fill=self.fill)
            elif image[TYPE] == LABEL:
                new_image = self.fill_rect(image.data, patches, fill=self.fill_label)
            image = new_image 

        return subject


    def fill_rect(self, tensor, patches, fill):
        per_channel = parse_per_channel(self.fill_per_channel, tensor.shape[0])        
        for ch in per_channel:
            if self.patch_per == 'channel':
                patches = self.get_patches(tensor.shape[1:])  
            for patch_i, patch in enumerate(patches):
                patch_slice = tuple(slice(pos_a, pos_a+shape_a) for pos_a, shape_a in zip(patch[0], patch[1]))
                tensor[ch][patch_slice] = self.get_fill_value(tensor[ch], patch_i, patch, fill) 
        return tensor

    @staticmethod
    def get_fill_value(tensor, patch_i, patch, fill):
        if fill == 'noise':
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            fill_value = torch.normal(mean, std, size=tuple(patch[1]))
        elif fill == 'random_image_values':
            fill_value = np.random.choice(tensor.flatten(), size=patch[1]) # very slow 
            # fill_value = fill[np.sum(n_elm[0:patch_i]): np.sum(n_elm[0:patch_i+1]) ].reshape(patch[1])
        elif isinstance(fill, Number):
            fill_value = fill
        elif isinstance(fill, callable):
            fill_value = fill(tensor, patch)
        else:
            assert f"Parameter 'fill' received invalid value: {fill}"
        return torch.tensor(fill_value)


    def get_patches(self, spatial_shape: np.ndarray) -> List[Tuple[TypeTripletInt, TypeTripletInt]]:
        # Calculate how many patches (including margin) fit into image (at most)
        patch_frame = self.patch_max+self.patch_margin 
        num_patches = np.floor_divide(spatial_shape, patch_frame)
        # offset = np.floor_divide(spatial_shape % patch_size, 2) # Start first patch so that patches are centered in image   
        offset = np.random.randint(spatial_shape % patch_frame+1) # Start first patch randomly between 0 and remaining space

        # Scale each patch within given bounds - [Patches, shape-(x,y,z)]
        patch_sizes =  np.asarray([np.random.randint(self.patch_min, self.patch_max+1) for _ in range(np.prod(num_patches))])

        # Calculate left(x),lower(y),front(z) corner of each patch 
        idx_enc = np.cumprod(np.array([1, *num_patches[:-1]])) # Index encoder 
        patch_pos = [ [ [ [
            tuple(offset[a]+v*patch_frame[a]+(patch_frame[a]-patch_sizes[idx,a])//2  for a, v in enumerate([ix,iy,iz]))
            for idx in (np.matmul([ix,iy,iz], idx_enc),)] # Note: only workaround to store variable idx here 
            for ix in range(0, num_patches[0])] 
            for iy in range(0, num_patches[1])] 
            for iz in range(0, num_patches[2])]

        patch_pos = np.asarray(patch_pos).reshape((-1, 3)) # [Patches, corner position-(x,y,z)],

        patches = np.stack([patch_pos, patch_sizes], axis=1) #  # [Patches, position/size, (x,y,z)]
        return patches # [patches, 2, 3]




       

class Brightness(RandomTransform):
    def __init__(self, scale, per_channel=True, **kwargs):
        super().__init__(**kwargs)
        self.per_channel = per_channel
        self.scale = scale 

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=True):
            if self.per_channel:
                new_data = []
                for ch_data in image.data:
                    new_data.append(ch_data * np.random.uniform(low=self.scale[0], high=self.scale[1]))
                image.set_data(np.stack(new_data))
            else:
                image.set_data(image.data * np.random.uniform(low=self.scale[0], high=self.scale[1]))
        return subject 


class RandomDisableChannel(RandomTransform):
    def __init__(self, channels, disable_per='subject', **kwargs):
        super().__init__(**kwargs)
        self.channels = np.array(channels)
        self.disable_per = disable_per

    def apply_transform(self, subject: Subject) -> Subject:
        if self.disable_per == 'subject':
            disable_channel = np.random.choice(self.channels)

        for image in subject.get_images(intensity_only=True):
            assert image.shape[0] >= len(self.channels), f"Image has only {image.shape[0]} channel, but {len(self.channels)} are disabled"
            if self.disable_per == 'image':
                disable_channel = np.random.choice(self.channels)
            new_data = []
            for ch, ch_data in enumerate(image.data):
                if ch == disable_channel:
                    new_data.append(ch_data*0)
                else:
                    new_data.append(ch_data)
            image.set_data(np.stack(new_data))
     
        return subject 


class SelectRandomChannel(Transform):
    """Select Random Channel"""
    def apply_transform(self, subject: Subject) -> Subject:
        images = subject.get_images(
            intensity_only=True, # WARNING: only apply to ScalarImages 
            include=self.include,
            exclude=self.exclude,
        )
        ch = torch.randint(images[0].shape[0], (1,)) # WARNING, assumes all images have same number of channels 
        for image in images:
            image.set_data(image.data[ch])
        return subject

class AddBlankChannel(RandomTransform):
    def __init__(self, channel, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=True):
            new_data = []
            for ch, ch_data in enumerate(image.data):
                if ch == self.channel:
                    new_data.append(ch_data*0)
                new_data.append(ch_data)
                if self.channel ==image.data.shape[0]:
                    new_data.append(ch_data*0)
            image.set_data(np.stack(new_data))
     
        return subject 


class SpatialTransform2(RandomAffine):
    """Equal to RandomAffine but allows that only two axis instead of three should be transformed in an isotropic manner"""
    def __init__(
            self,
            scales: TypeOneToSixFloat = 0.1,
            degrees: TypeOneToSixFloat = 10,
            translation: TypeOneToSixFloat = 0,
            isotropic: Union[bool, tuple] = False,
            center: str = 'image',
            default_pad_value: Union[str, float] = 'minimum',
            image_interpolation: str = 'linear',
            check_shape: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.isotropic = isotropic
        self._parse_scales_isotropic(scales, isotropic)
        self.scales = self.parse_params(scales, 1, 'scales', min_constraint=0)
        self.degrees = self.parse_params(degrees, 0, 'degrees')
        self.translation = self.parse_params(translation, 0, 'translation')
        if center not in ('image', 'origin'):
            message = (
                'Center argument must be "image" or "origin",'
                f' not "{center}"'
            )
            raise ValueError(message)
        self.center = center
        self.default_pad_value = _parse_default_value(default_pad_value)
        self.image_interpolation = self.parse_interpolation(
            image_interpolation)
        self.check_shape = check_shape

    def get_params(
            self,
            scales: TypeSextetFloat,
            degrees: TypeSextetFloat,
            translation: TypeSextetFloat,
            isotropic: Union[bool, tuple],
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaling_params = self.sample_uniform_sextet(scales)
        if isinstance(isotropic, bool) and isotropic:
            scaling_params.fill_(scaling_params[0])
        elif isinstance(isotropic, tuple): # (0,1),(0,2),(1,2) => x==y, x==z, y==z
            scaling_params[isotropic[0]] = scaling_params[isotropic[1]] 
        rotation_params = self.sample_uniform_sextet(degrees)
        translation_params = self.sample_uniform_sextet(translation)
        return scaling_params, rotation_params, translation_params
    
    def _parse_scales_isotropic(self, scales, isotropic):
        scales = to_tuple(scales)
        if isinstance(isotropic, bool):
            if isotropic and len(scales) in (3, 6):
                message = (
                    'If "isotropic" is True, the value for "scales" must have'
                    f' length 1 or 2, but "{scales}" was passed'
                )
                raise ValueError(message)
        elif isinstance(isotropic, tuple):
            if len(isotropic)!=2:
                raise ValueError("Parameter isotropic must be tuple with 3 booleans")
            if isotropic == (0,1): # x==y 
                if scales[0:2] != scales[2:4]:
                    raise ValueError("Scales differ in isotropic axis")
            elif isotropic == (0,2):
                if scales[0:2] != scales[4:6]:
                    raise ValueError("Scales differ in isotropic axis")
            elif isotropic == (1,2):
                if scales[2:4] != scales[4:6]:
                    raise ValueError("Scales differ in isotropic axis")


class PadFixed(tio.Pad):
    def apply_transform(self, subject: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            new_origin = nib.affines.apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            kwargs: Dict[str, Union[str, float]]
            if isinstance(self.padding_mode, Number):
                kwargs = {
                    'mode': 'constant',
                    'constant_values': self.padding_mode,
                }
            elif isinstance(image, tio.LabelMap): # FIX 
                kwargs = {
                    'mode': 'constant',
                    'constant_values': 0,
                }
            else:
                if self.padding_mode in ['maximum', 'mean', 'median', 'minimum']:
                    if self.padding_mode == 'maximum':
                        constant_values = image.data.min()
                    elif self.padding_mode == 'mean':
                        constant_values = image.data.to(torch.float).mean().to(image.data.dtype)
                    elif self.padding_mode == 'median':
                        constant_values = image.data.median()
                    elif self.padding_mode == 'minimum':
                        constant_values = image.data.min()
                    kwargs = {
                        'mode': 'constant',
                        'constant_values': constant_values,
                    }
                else:
                    kwargs = {'mode': self.padding_mode}
            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image.data, paddings, **kwargs)  # type: ignore[call-overload]  # noqa: E501
            image.set_data(torch.as_tensor(padded))
            image.affine = new_affine
        return subject


class CropOrPadFixed(tio.CropOrPad):
    """Fixed version of TorchIO CropOrPad: 
         * Pads with zeros for LabelMaps independent of padding mode (eg. don't pad with mean)
       Changes: 
         * Pads with global (not per axis) 'maximum', 'mean', 'median', 'minimum' if any of these padding modes were selected"""
    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            pad = PadFixed(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)  # type: ignore[assignment]
        return subject

class RandomCropOrPad(CropOrPadFixed):
    """CropOrPad but bounding box position is set randomly."""
    # Random margins to crop or pad 
    @staticmethod
    def _get_six_bounds_parameters( parameters: np.ndarray) :
        result = []
        for number in parameters:
            ini = np.random.randint(low=0, high=number+1)
            fin = number-ini
            result.extend([ini, fin])
        return tuple(result)



class CropOrPadNone(CropOrPadFixed):
    """CropOrPad enables axis not to be changed by setting to None in target_shape """
    def __init__(
            self,
            target_shape: Union[int, TypeTripletInt, None] = None,
            padding_mode: Union[str, float] = 0,
            mask_name: Optional[str] = None,
            labels: Optional[Sequence[int]] = None,
            **kwargs
            ):

            # WARNING: Ugly workaround to allow None values
            if target_shape is not None:
                self.original_target_shape = to_tuple(target_shape, length=3)
                target_shape = [1 if t_s is None else t_s for t_s in target_shape]
            super().__init__(target_shape, padding_mode, mask_name, labels, **kwargs)

    def apply_transform(self, subject: Subject):
        # WARNING: This makes the transformation subject dependent - reverse transformation must be adapted 
        if self.target_shape is not None:
            self.target_shape = [s_s if t_s is None else t_s for t_s, s_s in zip(self.original_target_shape, subject.spatial_shape)]
        return super().apply_transform(subject=subject)
        




class ContrastAugmentationTransform(RandomTransform):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_channel=1, **kwargs):
        super().__init__(**kwargs)
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel


    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=True):
            new_data = augment_contrast(image.data,
                            contrast_range=self.contrast_range,
                            preserve_range=self.preserve_range,
                            per_channel=self.per_channel,
                            p_per_channel=self.p_per_channel)
            image.set_data(new_data)

        return subject


class SimulateLowResolutionTransform(RandomTransform):
    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1, channels=None, order_downsample=1, order_upsample=0, ignore_axes=None, **kwargs):
        super().__init__(**kwargs)
        self.zoom_range = zoom_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.channels = channels
        self.order_downsample = order_downsample
        self.order_upsample = order_upsample        
        self.ignore_axes = ignore_axes


    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=True):
            new_data = augment_linear_downsampling_scipy(image.data.numpy(), zoom_range=self.zoom_range, per_channel=self.per_channel, p_per_channel=self.per_channel,
                                      channels=self.channels, order_downsample=self.order_downsample, order_upsample=self.order_upsample, ignore_axes=self.ignore_axes)
            image.set_data(new_data)

        return subject



class Inverse(Transform):
    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=True):
            image.set_data(-image.data)
        return subject 

class Trans3Dto2D(Transform):
    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):
            shape = image.data.shape  # [C, W,H,D]
            image.set_data(np.expand_dims(image.data.reshape((-1, *shape[1:3])), -1))
            subject['_3dto2d_depth'] = shape[-1]
        return subject 


class Trans2Dto3D(Transform):
    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):
            shape = image.data.shape  # [C, W,H,D]
            channels = int(shape[0]/subject['_3dto2d_depth'])
            image.set_data(np.squeeze(image.data, -1).reshape((channels, *shape[1:3],-1)) )
        return subject 

class Resample2(SpatialTransform):
    """Resample based on nnUNet implementation."""
    def __init__(self, target_spacing, **kwargs):
        super().__init__(**kwargs)
        self.target_spacing = target_spacing
        self.args_names = (
            'target_spacing',
        )
        
    
    def apply_transform(self, subject: Subject) -> Subject:
        subject['_org_spacing'] = subject.spacing
        subject['_org_spatial_shape'] = subject.spatial_shape
        for image in subject.get_images(intensity_only=False):
            target_spacing = self.target_spacing[::-1]
            original_spacing  = image.spacing[::-1]
            data = np.swapaxes(image.data.numpy(), 1, -1)
            image_data = data if image[TYPE] == INTENSITY else None 
            seg_data = None if image[TYPE] == INTENSITY else data 
            image_data, seg_data = resample_patient(image_data, seg_data, original_spacing, target_spacing) # Order [(C),D,H,W]
            data = image_data if image[TYPE] == INTENSITY else seg_data 
            data = np.swapaxes(data, 1, -1)
            image.set_data(data)
            image.affine = get_affine2(self.target_spacing,  image.direction, image.origin, lps2ras=False)
        return subject 

    def is_invertible(self):
        return True
    
    def inverse(self):
        return Resample2Inverse()

class ResampleTio(Resample):     
    """Resample is based on the TorchIO implementation, but is reversible"""
    def apply_transform(self, subject: Subject) -> Subject:
        subject['_org_spacing'] = subject.spacing
        subject['_org_spatial_shape'] = subject.spatial_shape
        return super().apply_transform(subject)

    def is_invertible(self):
        return True
    
    def inverse(self):
        return ResampleTio2Inverse()




class RandomResample(SpatialTransform):
    """Random resample augmentation"""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    
    def apply_transform(self, subject: Subject) -> Subject:
        subject['_org_spacing'] = subject.spacing
        subject['_org_spatial_shape'] = subject.spatial_shape

        xy = torch.rand(1).item()*0.5+0.5 # [0.5, 1.0]
        z = np.random.choice([1.5, 2, 2.5, 3])
        target_spacing =  np.array([xy, xy, z]) 

        return Resample(target_spacing)(subject)

    def is_invertible(self):
        return True
    
    def inverse(self):
        # return Resample2Inverse()
        return ResampleTio2Inverse()

class Resample2Inverse(SpatialTransform):
    """Reverse function for Resample2"""
    def apply_transform(self, subject: Subject) -> Subject:
        return Compose([Resample2(subject['_org_spacing']), CropOrPad(subject['_org_spatial_shape'])])(subject)
        

class ResampleTio2Inverse(SpatialTransform):
    """Reverse function for ResampleTio"""
    def apply_transform(self, subject: Subject) -> Subject:
        return Compose([Resample(subject['_org_spacing']), CropOrPad(subject['_org_spatial_shape'])])(subject)        


        

class HistogramStandardization2(HistogramStandardization):
    """TorchIO-HistogramStandardization which allows the input of images instead of paths and skips empty masks."""
    DEFAULT_CUTOFF = 0.01, 0.99
    STANDARD_RANGE = 0, 100

    @classmethod
    def train(
            cls,
            images: Sequence,
            mask_name: str = None,
            cutoff: Optional[Tuple[float, float]] = None,
            output_path: Optional[TypePath] = None,
            ) -> np.ndarray:
     
        quantiles_cutoff = cls.DEFAULT_CUTOFF if cutoff is None else cutoff
        percentiles_cutoff = 100 * np.array(quantiles_cutoff)
        percentiles_database = []
        percentiles = cls._get_percentiles(percentiles_cutoff)
        for i, item in enumerate(tqdm(images)):
            array = item['source'].numpy()
            if mask_name is not None:
                mask = item[mask_name].numpy() > 0
                if not mask.any():
                    print("WARNING: Skipping because no valid label in mask")
                    continue
            if len(array[mask]) < 20:
                print("WARNING: Percentile calculation only based on less than 20 values")
            percentile_values = np.percentile(array[mask], percentiles)
            if percentile_values[0] == percentile_values[-1]:
                raise Exception("Percentiles should not be equal")
            percentiles_database.append(percentile_values)
        percentiles_database = np.vstack(percentiles_database)
        mapping = cls._get_average_mapping(percentiles_database)

        if output_path is not None:
            output_path = Path(output_path).expanduser()
            extension = output_path.suffix
            if extension == '.txt':
                modality = 'image'
                text = f'{modality} {" ".join(map(str, mapping))}'
                output_path.write_text(text)
            elif extension == '.npy':
                np.save(output_path, mapping)
        return mapping

    @classmethod 
    def _get_percentiles(cls, percentiles_cutoff: Tuple[float, float]) -> np.ndarray:
        quartiles = np.arange(25, 100, 25).tolist()
        deciles = np.arange(10, 100, 10).tolist()
        all_percentiles = list(percentiles_cutoff) + quartiles + deciles
        percentiles = sorted(set(all_percentiles))
        return np.array(percentiles)

    @classmethod 
    def _get_average_mapping(cls, percentiles_database: np.ndarray) -> np.ndarray:
        """Map the landmarks of the database to the chosen range.

        Args:
            percentiles_database: Percentiles database over which to perform the
                averaging.
        """
        # Assuming percentiles_database.shape == (num_data_points, num_percentiles)
        pc1 = percentiles_database[:, 0]
        pc2 = percentiles_database[:, -1]
        s1, s2 = cls.STANDARD_RANGE
        slopes = (s2 - s1) / (pc2 - pc1)
        slopes = np.nan_to_num(slopes)
        intercepts = np.mean(s1 - slopes * pc1)
        num_images = len(percentiles_database)
        final_map = slopes.dot(percentiles_database) / num_images + intercepts
        return final_map


class LambdaSubject(Transform):
    """Lambda function that is applied on Subject"""
    def __init__(
        self,
        function: TypeCallable,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.function = function
        self.args_names = ('function',)

    def apply_transform(self, subject: Subject) -> Subject:
        return self.function(subject)



class ToOrientation(SpatialTransform):
    """Generalization of TorchIO-ToCanonical """
    def __init__(self, target_orientation, **kwargs):
        self.target_orientation = target_orientation 
        super().__init__(**kwargs)
        self.args_names = ('target_orientation',)

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            subject[f'_org_orientation_{image_name}'] = image.orientation
            self._reorient(image, self.target_orientation)
        return subject
    
    @classmethod
    def _reorient(cls, image, target_orientation):
        if image.orientation == tuple(target_orientation):
            return
        affine = image.affine
        array = image.numpy()[np.newaxis]  # (1, C, W, H, D)
        # NIfTI images should have channels in 5th dimension
        array = array.transpose(2, 3, 4, 0, 1)  # (W, H, D, 1, C)
        nii = nib.Nifti1Image(array, affine)
        
        original_ornt = nib.io_orientation(affine)
        target_ornt = nib.orientations.axcodes2ornt(target_orientation)
        transform = nib.orientations.ornt_transform(original_ornt, target_ornt)
        reoriented = nii.as_reoriented(transform)

        # https://nipy.org/nibabel/reference/nibabel.dataobj_images.html#nibabel.dataobj_images.DataobjImage.get_data
        array = np.asanyarray(reoriented.dataobj)
        # https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
        array = array.copy()
        array = array.transpose(3, 4, 0, 1, 2)  # (1, C, W, H, D)
        image.set_data(torch.as_tensor(array[0]))
        image.affine = reoriented.affine

    def is_invertible(self):
        return True
    
    def inverse(self):
        return ToOrientationInverse()

class ToOrientationInverse(SpatialTransform):    
    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            target_orientation = subject[f'_org_orientation_{image_name}']
            ToOrientation._reorient(image, target_orientation)
        return subject

