import logging
from pathlib import Path 
import json 

import torchio as tio
import SimpleITK as sitk
import numpy as np 

from breaststudies.augmentation import  ZNormalization, CropOrPadFixed
from breaststudies.data import BaseDataset
from breaststudies.utils import get_affine

logger = logging.getLogger(__name__)

class BreastDataset(BaseDataset):
    path_root_default = Path('/home/gustav/Documents/datasets/BreastDataset/Gustav')

    default_target = 'tissue'
    default_target_files = {'tissue': {'target':'mask_tissue.nii.gz', 'mask':'mask_breast.nii.gz'}, 'breast': {'target':'mask_breast.nii.gz'}}
    default_source_files = {'tissue': {'source': [ 'Dyn_0.nii', 'T2_resampled.nii', 'Sub.nii' ]}, 'breast':{'source':['Dyn_0.nii', 'T2_resampled.nii']} } 

    label_dict_tissue = {'Background':0, 'FGT':1, 'Fibroadenoma':2, 'Carcinoma':3, 'DCIS':4} 
    label_dict_breast = {'Background':0, 'Breast':1, 'Implant':2} 

    default_roi = (256, 256, 32)

    # --------------------------------- Preprocessing ---------------------------------------
    # Is applied to individual items (eg. Left/Right or Slice) i.e. after 'series2items' 
    default_item_trans = tio.Compose([
        # ZNormalization(),
        # CropOrPadFixed(default_roi, padding_mode='minimum') 
    ])

    # Is applied to the entire 3D volume 
    default_series_trans = tio.ToCanonical()
    
  

    def _post_init(self):
        target = self.kwargs.get('target', None)
        if (target is not None) and ('target_files' not in self.kwargs):
            self.kwargs['target_files'] = self.default_target_files[target]
        if (target is not None) and ('source_files' not in self.kwargs):
            self.kwargs['source_files'] = self.default_source_files[target]
    
    
    @classmethod
    def init_crawler(cls, path_root=None, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        return ((path_dir.relative_to(path_root),) for path_dir in Path(path_root).iterdir() if path_dir.is_dir())


    @classmethod
    def item_pointer2uid(cls, item_pointer, path_root=None, id_type='item', **kwargs): 
        """Returns identifier for each item.

        Args:
            item_pointer (object): Unique pointer (e.g. Path or (Path, Slice)) to each item 
            path_root (Path, optional): Path to root directory 
            id_type (str, optional): Specifies the id-type e.g. case or study 
        """
        path_root = cls.path_root_default if path_root is None else Path(path_root)
        path_item = path_root/Path(item_pointer[0])

        if id_type=="patient":
            dicom_info = cls.load_dicom_tags(item_pointer, path_root=path_root, **kwargs)
            return dicom_info["PatientID"]  # Hint: PatientID (0010,0020) 
        if id_type == "study":  
            raise NotImplementedError
        elif id_type == "series": 
            return item_pointer[0].name # e.g. 90xxxxx
        elif id_type == "item": 
            return item_pointer[0].name # e.g. 90xxxxx
       

    @classmethod
    def load_series(cls, item_pointer, path_root, **kwargs):
        "Load (image) data and return as dict"
        source_files = kwargs.get('source_files',  cls.default_source_files[cls.default_target])
        target_files = kwargs.get('target_files', cls.default_target_files[cls.default_target])
        path_item_source = Path(path_root)/item_pointer[0]
        path_item_target = {name:Path(kwargs.get('path_root_target', {}).get(name, path_root))/item_pointer[0] for name in target_files.keys()} # Target files (masks) might be placed under a different folder 
        series_id = cls.item_pointer2uid(item_pointer, path_root, 'series')

        spacing = None 

        if kwargs.get('raw', False):
            sources = dict({name:(sitk.ReadImage(str(path_item_source/filename), sitk.sitkFloat32) if (path_item_source/filename).is_file() else None)  for name, filename in source_files.items()})
            targets = dict({name:(sitk.ReadImage(str(path_item_target[name]/filename), sitk.sitkUInt8)   if (path_item_target[name]/filename).is_file() else None)  for name, filename in target_files.items()})
            dicomtags = cls.load_dicom_tags(item_pointer, path_root, **kwargs)
          
            return {'uid': series_id, **sources, **targets, **dicomtags}
    

        # ---------- Load source(s) ------------   
        sources = {}
        for source_name, channel_files in source_files.items():
            try: 
                channels = []
                for channel_file in channel_files:
                    ch_nii = sitk.ReadImage(str(path_item_source/channel_file), sitk.sitkFloat32)
                    affine = get_affine(ch_nii)
                    ch_np = sitk.GetArrayFromImage(ch_nii).transpose() # [W, H, D], WARNING TorchIO opposite PyTorch order   
                    channels.append(ch_np)
                channels = np.stack(channels, axis=0)
                source = tio.ScalarImage(tensor=channels, affine=affine) # WARNING: generally affine not equal across different images 
                spacing = source.spacing 
                sources[source_name] = source
            except Exception as e:
                logger.warning(f"Could not load {source_name} because: {e}")


        # ---------- Load target ---------------
        targets = {} 
        for name, target_file in target_files.items():
            target = sitk.ReadImage(str(path_item_target[name]/target_file), sitk.sitkUInt8)
            affine = get_affine(target)
            target = sitk.GetArrayFromImage(target).transpose() # [W, H, D], WARNING TorchIO opposite PyTorch order
            target = target.astype(np.float32) # Will be casted as long - prevents error when using num_workers>0 and stack()
            target = target[None] # [C, W, H, D]
            target = kwargs.get('manipulate_label_func', lambda x:x[0])((target, name, sources)) # eg. lambda x: np.where(x[0]>1, 0, 1) if x[1] == 'target' else x[0]        
            targets[name] = tio.LabelMap(tensor=target, affine=affine)
            spacing = targets[name].spacing 
        

        # ----------- Load Metadata -------------
        dicomtags = cls.load_dicom_tags(item_pointer, path_root, **kwargs) if kwargs.get('load_dicomtags', False) else {}
        spacing = {'spacing':np.array(spacing, dtype=np.float32)} if spacing is not None else {} 
       

        return tio.Subject({'uid': series_id, **sources, **targets, **spacing, **dicomtags }) 

         
    
    @classmethod
    def tio2torch(self, series):
        # Transform TorchIO Subject to PyTorch (Tensor) and Shape-Order 
        return {key: val.data.swapaxes(1,-1) if isinstance(val, tio.Image) else val  for key,val in series.items()}
    
    @classmethod
    def tio2numpy(self, series):
        # Transform TorchIO Subject to PyTorch (Tensor) and Shape-Order 
        return {key: val.numpy().swapaxes(1,-1) if isinstance(val, tio.Image) else val  for key,val in series.items()}
    
    @classmethod
    def tio2sitk(self, series):
        # Transform TorchIO Subject to PyTorch (Tensor) and Shape-Order 
        return {key: val.as_sitk() if isinstance(val, tio.Image) else val  for key,val in series.items()}

    @classmethod
    def item2out(cls, item, **kwargs):
        out_format = kwargs.get('out_format', 'torch')
        if out_format == 'torch':
            return cls.tio2torch(item)
        if out_format == 'numpy':
            return cls.tio2numpy(item)
        elif out_format == 'sitk':
            return cls.tio2sitk(item)
        elif out_format == 'tio':
            return item 

    @property
    def labels(self):
        "Return a dict of label names and their label values e.g. {'Lesion A':1, 'Lesion B':2} "
        if self.kwargs.get('target', 'tissue') == 'tissue':
            return self.label_dict_tissue
        else: 
            return self.label_dict_breast
      
    
    @property
    def label_fcts(self):
        "Return a dict of label names (and combinations) and their mask function e.g. {'All Lesions':lambda x:x>=1} "
        if self.kwargs.get('target', 'tissue') == 'tissue': 
            return {'Background': lambda x:x==0,
                    'FGT': lambda x:x==1, 
                    'Fibroadenoma': lambda x:x==2, 
                    'Carcinoma': lambda x:x==3, 
                    'DCIS': lambda x:x==4,
                    'FGT or Carcinoma': lambda x:x==5,
                    'B3 Lesion': lambda x:x==6} 
        else:
            return {'Background': lambda x:x==0,
                    'Breast': lambda x:x==1, 
                    'Implant': lambda x:x==2 } 

    
    @classmethod
    def load_dicom_tags(cls, item_pointer, path_root=None, **kwargs):
        path_item = Path(path_root)/item_pointer[0]
        file_name = kwargs.get('json_file', 'Dyn.json')
        with open(path_item/file_name, 'r') as f:
            dicom_tags = json.load(f)
        return dicom_tags

    @classmethod
    def apply_crop(cls, subject, crop_dict):
        subjects = {} 
        for crop_name, trans in crop_dict.items():
            subject_side = trans(subject)
            subject_side['uid'] = subject['uid']+'_'+crop_name
            subjects[crop_name] = subject_side
        return subjects 


 
class BreastDatasetLR(BreastDataset):
    @classmethod
    def init_crawler(cls, path_root, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        return ((path_dir.relative_to(path_root),side) for path_dir in Path(path_root).iterdir() for side in ['left', 'right'] if path_dir.is_dir())
    
    @classmethod
    def item_pointer2uid(cls, item_pointer, path_root=None, id_type='item'): 
        if id_type=='item':
            return  '_'.join([str(elem) for elem in item_pointer])
        else:
            return BreastDataset.item_pointer2uid(item_pointer, path_root=path_root, id_type=id_type)


    @classmethod
    def get_lr_crop(cls, subject, **kwargs): 
        shape = subject.spatial_shape # [W, H, D]
        crop_dict = {'left':tio.Crop((shape[0]//2, 0, 0, 0, 0, 0)),
                    'right':tio.Crop((0, shape[0]//2, 0, 0, 0, 0)) }
        return crop_dict 

    @classmethod
    def get_lr_breast_crop(cls, subject, **kwargs):
        crop_dict = cls.get_lr_crop(subject)
        target_shape = kwargs.get('target_shape', None) # If set, it's used here as a minimum shape (extend bbox if target_shape>mask_shape)
        patches = {}
        for side in ['left', 'right']:
            if target_shape is not None:
                mask_bbox = np.diff(CropOrPadFixed._bbox_mask(crop_dict[side](subject['mask']).numpy()[0]), axis=0)[0]
                target_shape = tuple(max(x, y) if x is not None else y for x, y in zip(target_shape, mask_bbox))
            patches[side] = tio.Compose([
                crop_dict[side],
                CropOrPadFixed(target_shape, mask_name='mask', labels=(1,), padding_mode=0)
            ])
        return patches 

    @classmethod
    def series2items(cls, item_pointer, series, **kwargs):
        # ------- Get dict that specifies crop region  
        if kwargs.get('target') == 'tissue':
            crop_dict = cls.get_lr_breast_crop(series, **kwargs)
        else:
            crop_dict = cls.get_lr_crop(series, **kwargs)

        series_crops = cls.apply_crop(series, crop_dict) # Also adapts the UID
        return { (*item_pointer, crop_name):crop_series for crop_name, crop_series in series_crops.items()}
  

       
class BreastDataset2D(BreastDataset):
 
    @classmethod
    def init_crawler(cls, path_root, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        return ( (path_dir.relative_to(path_root), str(slice_n)) 
                for path_dir in Path(path_root).iterdir() 
                for slice_n in range(cls.load_dicom_tags((path_dir,), path_root, **kwargs)['_NumberOfSlices'])
                if path_dir.is_dir()   )

    @classmethod
    def _get_slice_crop(cls, slices, **kwargs):
        return {str(slice_i):tio.Crop((0,0, 0,0, slice_i,slices-(slice_i+1)))  for slice_i in range(slices) }

    @classmethod
    def series2items(cls, item_pointer, series, **kwargs):
        crop_dict = cls._get_slice_crop(series.spatial_shape[-1], **kwargs)
        series_crops = cls.apply_crop(series, crop_dict)
        return { (*item_pointer, crop_name):crop_series for crop_name, crop_series in series_crops.items()}

    @classmethod
    def item_pointer2uid(cls, item_pointer, path_root=None, id_type='item'): 
        if id_type=='item':
            return str(item_pointer[0])+'_'+str(item_pointer[1])
        else:
            return BreastDataset.item_pointer2uid(item_pointer, path_root=path_root, id_type=id_type)


class BreastDatasetLR2D(BreastDatasetLR, BreastDataset2D):
    @classmethod
    def init_crawler(cls, path_root, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        def slice_crawler_func(cls, path_root, path_dir, side, **kwargs):
            return cls.load_dicom_tags((path_dir,), path_root, **kwargs)['_NumberOfSlices']

        slice_crawler_func = kwargs.get('slice_crawler_func',slice_crawler_func) 
        # uids_exclude = [path_file.stem.split('_')[0] for path_file in Path('/mnt/hdd/datasets/breast/Pix2Pix/Pix2Pix_LowDoseOnly2/train_A').iterdir()]
        # dirs = [path_dir for path_dir in  Path(path_root).iterdir() if path_dir.name not in uids_exclude ]
        return ( (path_dir.relative_to(path_root), side, str(slice_n) ) 
                for path_dir in Path(path_root).iterdir()
                for side in ['left', 'right']
                for slice_n in range(slice_crawler_func(cls, path_root, path_dir, side, **kwargs))
                if path_dir.is_dir()   )

    @classmethod
    def series2items(cls, item_pointer, series, **kwargs):
        items_lr =  BreastDatasetLR.series2items(item_pointer[0:1], series, **kwargs)
        items = {}
        for sub_item_pointer, sub_series in items_lr.items():
            items.update(BreastDataset2D.series2items(sub_item_pointer, sub_series, **kwargs))
        return items 



class BreastUKADataset(BreastDataset):
    path_root_default = Path('/mnt/hdd/datasets/breast/UKA/UKA_2021_05_25/')
    default_target_files = {'tissue': {'target':'mask_tissue_3dunet.nii.gz', 'mask':'mask_breast_3dunet.nii.gz'},
                           'breast': {'target':'mask_breast_3dunet.nii.gz'}}

    @classmethod
    def manipulate_label_func(cls, x):
        target,name, sources = x 
        if (name == 'target'):
            # Option - simplified 
            target[target==5] = 1 # Assume normal breast tissue in transition area between carcinoma and breast tissue (label 5)
            target[target==6] = 0 # Assume "background" for radial scar (label 6) (very difficult for expert to differentiate between those classes, potential to be malignant)

            # Option - only FGT 
            # target[target>1] = 0 # "Remove" everything that is not FGT 

            # Option - only Carcinoma 
            # target[target !=3] = 0 # Everything that is no a Carcinoma, remove
            # target[target ==3] = 1 # Relabel Carcinoma as Label 1

            # Option - combine DCIS and Carcinoma 
            target[target==4] = 3
            
        elif (name == 'mask') and len(sources):
            ch = sources['source'].shape[0] 
            target = target>0 # Don't separate between subcategories (eg. implants)
            target = np.tile(target, (ch,1,1,1)) # Number of mask-channels must match number of source channels
        return target 
        

class BreastUKADatasetLR(BreastUKADataset, BreastDatasetLR):
    pass 

class BreastUKADataset2D(BreastUKADataset, BreastDataset2D):
    pass 

class BreastUKADatasetLR2D(BreastUKADataset, BreastDatasetLR2D):
    pass 

# --------------------------------------------- DUKE -----------------------------------------------------------------
class BreastDUKEDataset(BreastDataset):
    path_root_default = Path('/mnt/hdd/datasets/breast/DUKE/dataset')
    default_target_files = {'tissue': {'target':'mask_tissue_3dunet.nii.gz', 'mask':'mask_breast_3dunet.nii.gz'}, 
                            'breast': {'target':'mask_breast_3dunet.nii.gz'}}
    default_source_files = {'tissue': {'source': ['T1.nii.gz', 'sub_resampled.nii']}, 'breast':{'source':['T1.nii.gz']} } 
    
    

    @classmethod
    def load_dicom_tags(cls, item_pointer, path_root=None, **kwargs):
        path_item = Path(path_root)/item_pointer[0]
        file_name = kwargs.get('json_file', 'pre.json')
        with open(path_item/file_name, 'r') as f:
            dicom_tags = json.load(f)
        return dicom_tags


class BreastDUKEDatasetLR(BreastDUKEDataset, BreastDatasetLR):
    pass

class BreastDUKEDataset2D(BreastDUKEDataset, BreastDataset2D):
    pass 

class BreastDUKEDatasetLR2D(BreastDUKEDataset, BreastDatasetLR2D):
    pass 

class BreastDUKESubsetDataset(BreastDUKEDataset):
    default_target_files = {'tissue': {'target':'mask_tissue.nii.gz', 'mask':'mask_breast.nii.gz'}, 
                            'breast': {'target':'mask_breast.nii.gz'}}
    path_root_default = Path('/home/gustav/Documents/datasets/BreastDataset/DUKESubset')

class BreastDUKESubsetDatasetLR(BreastDUKESubsetDataset, BreastDUKEDatasetLR):
    pass 

class BreastDUKESubsetDataset2D(BreastDUKESubsetDataset, BreastDUKEDataset2D):
    pass 

class BreastDUKESubsetDatasetLR2D(BreastDUKESubsetDataset, BreastDUKEDatasetLR2D):
    pass 

# ------------------------------- BREAST-DIAGNOSIS -------------------------------------------------------------
class BreastDIAGNOSISDataset(BreastDataset):
    path_root_default = Path('/mnt/hdd/datasets/breast/BREAST-DIAGNOSIS/dataset')
    default_target_files = {'tissue': {'target':'mask_tissue_3dunet.nii.gz', 'mask':'mask_breast_3dunet.nii.gz'}, 
                            'breast': {'target':'mask_breast_3dunet.nii.gz'}}
    default_source_files = {'tissue': {'source': ['T2.nii.gz', 'sub_resampled.nii']}, 'breast':{'source':['T2.nii.gz']} }



    @classmethod
    def init_crawler(cls, path_root=None, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        return ((path_sub_dir.relative_to(path_root),) for path_dir in Path(path_root).iterdir() if path_dir.is_dir() 
                for path_sub_dir in Path(path_dir).iterdir() if path_sub_dir.is_dir()  )
 

    @classmethod
    def item_pointer2uid(cls, item_pointer, path_root=None, id_type='item', **kwargs): 
        """Returns identifier for each item.

        Args:
            item_pointer (object): Unique pointer (e.g. Path or (Path, Slice)) to each item 
            path_root (Path, optional): Path to root directory 
            id_type (str, optional): Specifies the id-type e.g. case or study 
        """
        if id_type=="patient":
            dicom_info = cls.load_dicom_tags(item_pointer, path_root=path_root, **kwargs)
            return dicom_info["PatientID"]  # Hint: PatientID (0010,0020) 
        if id_type == "study":  
            return item_pointer[0].parts[0] # e.g. 0001
        elif id_type == "series": 
            return '_'.join(item_pointer[0].parts) 
        elif id_type == "item": 
            return '_'.join(item_pointer[0].parts) # e.g. 0001_08-12-2008
        else:
            raise "Unknown id_type"

    @classmethod
    def load_dicom_tags(cls, item_pointer, path_root=None, **kwargs):
        path_root = cls.path_root_default if path_root is None else Path(path_root)
        path_item = Path(path_root)/item_pointer[0]
        file_name = kwargs.get('json_file', 'BLISS.json')
        path_file = path_item/file_name
        path_file = path_file if path_file.is_file() else path_item/'T2.json'
        path_file = path_file if path_file.is_file() else path_item/'STIR.json'
        with open(path_file, 'r') as f:
            dicom_tags = json.load(f)
        return dicom_tags

class BreastDIAGNOSISDatasetLR(BreastDIAGNOSISDataset, BreastDatasetLR):
    @classmethod
    def init_crawler(cls, path_root=None, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        return ((path_sub_dir.relative_to(path_root), side) for path_dir in Path(path_root).iterdir() if path_dir.is_dir() 
                for path_sub_dir in Path(path_dir).iterdir() if path_sub_dir.is_dir() for side in ['left', 'right']   ) 

class BreastDIAGNOSISDataset2D(BreastDIAGNOSISDataset, BreastDataset2D):
    pass 

class BreastDIAGNOSISDatasetLR2D(BreastDIAGNOSISDataset, BreastDatasetLR2D):
    pass 

class BreastDIAGNOSISSubsetDataset(BreastDIAGNOSISDataset):
    default_target_files = {'tissue': {'target':'mask_tissue.nii.gz', 'mask':'mask_breast.nii.gz'}, 
                            'breast': {'target':'mask_breast.nii.gz'}}
    path_root_default = Path('/home/gustav/Documents/datasets/BreastDataset/BREAST-DIAGNOSIS-Subset')

class BreastDIAGNOSISSubsetDatasetLR(BreastDIAGNOSISSubsetDataset, BreastDIAGNOSISDatasetLR):
    pass

class BreastDIAGNOSISSubsetDataset2D(BreastDIAGNOSISSubsetDataset, BreastDIAGNOSISDataset2D):
    pass

class BreastDIAGNOSISSubsetDatasetLR2D(BreastDIAGNOSISSubsetDataset, BreastDIAGNOSISDatasetLR2D):
    pass

# ------------------------------------------------- TCGA -------------------------------------------
class BreastTCGABRCADataset(BreastDataset):
    path_root_default = Path('/mnt/hdd/datasets/breast/TCGA-BRCA/dataset') 
    default_target_files = {'tissue': {'target':'mask_tissue_3dunet.nii.gz', 'mask':'mask_breast_3dunet.nii.gz'}, 
                           'breast': {'target':'mask_breast_3dunet.nii.gz'}}
    default_source_files = {'tissue': {'source': ['T1.nii.gz', 'sub.nii']}, 'breast':{'source':['T1.nii.gz']} }


    @classmethod
    def init_crawler(cls, path_root=None, **kwargs):
        """Return an iterator over all items e.g. path_root.rglob('*.jpg')"""
        return ((path_sub_dir.relative_to(path_root),) for path_dir in Path(path_root).iterdir() if path_dir.is_dir() 
                for path_sub_dir in Path(path_dir).iterdir() if path_sub_dir.is_dir()  )


    @classmethod
    def item_pointer2uid(cls, item_pointer, path_root=None, id_type='item', **kwargs): 
        """Returns identifier for each item.

        Args:
            item_pointer (object): Unique pointer (e.g. Path or (Path, Slice)) to each item 
            path_root (Path, optional): Path to root directory 
            id_type (str, optional): Specifies the id-type e.g. case or study 
        """
        if id_type=="patient":
            dicom_info = cls.load_dicom_tags(item_pointer, path_root=path_root, **kwargs)
            return dicom_info["PatientID"]  # Hint: PatientID (0010,0020) 
        if id_type == "study":  
            return item_pointer[0].parts[0] # e.g. 0001
        elif id_type == "series": 
            return '_'.join(item_pointer[0].parts) 
        elif id_type == "item": 
            return '_'.join(item_pointer[0].parts) # e.g. 0001_08-12-2008
        else:
            raise "Unknown id_type"

    @classmethod
    def load_dicom_tags(cls, item_pointer, path_root=None, **kwargs):
        path_root = cls.path_root_default if path_root is None else Path(path_root)
        path_item = Path(path_root)/item_pointer[0]
        file_name = kwargs.get('json_file', 'pre.json')        
        with open(path_item/file_name, 'r') as f:
            dicom_tags = json.load(f)
        return dicom_tags

class BreastTCGABRCADatasetLR(BreastDatasetLR, BreastTCGABRCADataset):
    pass 




def BreastDatasetCreator(dataset_name, cohort, use_2d=False, lateral='bilateral', **kwargs):
    if dataset_name == 'uka':
        if cohort == 'entire':
            if (use_2d == True) and (lateral == 'bilateral'):
                return BreastUKADataset2D(**kwargs)
            elif (use_2d == True) and (lateral == 'unilateral'):
                return BreastUKADatasetLR2D(**kwargs)
            elif (use_2d == False) and (lateral == 'bilateral'):
                return BreastUKADataset(**kwargs)
            elif (use_2d == False) and (lateral == 'unilateral'):
                return BreastUKADatasetLR(**kwargs)
        elif cohort == 'subset':
            if (use_2d == True) and (lateral == 'bilateral'):
                return BreastDataset2D(**kwargs)
            elif (use_2d == True) and (lateral == 'unilateral'):
                return BreastDatasetLR2D(**kwargs)
            elif (use_2d == False) and (lateral == 'bilateral'):
                return BreastDataset(**kwargs)
            elif (use_2d == False) and (lateral == 'unilateral'):
                return BreastDatasetLR(**kwargs)
        else:
            raise Exception("Cohort name unknown.")
    elif dataset_name == 'duke':
        if cohort == 'entire':
            if (use_2d == True) and (lateral == 'bilateral'):
                return BreastDUKEDataset2D(**kwargs)
            elif (use_2d == True) and (lateral == 'unilateral'):
                return BreastDUKEDatasetLR2D(**kwargs)
            elif (use_2d == False) and (lateral == 'bilateral'):
                return BreastDUKEDataset(**kwargs)
            elif (use_2d == False) and (lateral == 'unilateral'):
                return BreastDUKEDatasetLR(**kwargs)
        elif cohort == 'subset':
            if (use_2d == True) and (lateral == 'bilateral'):
                return BreastDUKESubsetDataset2D(**kwargs)
            elif (use_2d == True) and (lateral == 'unilateral'):
                return BreastDUKESubsetDatasetLR2D(**kwargs)
            elif (use_2d == False) and (lateral == 'bilateral'):
                return BreastDUKESubsetDataset(**kwargs)
            elif (use_2d == False) and (lateral == 'unilateral'):
                return BreastDUKESubsetDatasetLR(**kwargs)
        else:
            raise Exception("Cohort name unknown.")
    elif dataset_name == 'breast-diagnosis':
        if cohort == 'entire':
            if (use_2d == True) and (lateral == 'bilateral'):
                return BreastDIAGNOSISDataset2D(**kwargs)
            elif (use_2d == True) and (lateral == 'unilateral'):
                return BreastDIAGNOSISDatasetLR2D(**kwargs)
            elif (use_2d == False) and (lateral == 'bilateral'):
                return BreastDIAGNOSISDataset(**kwargs)
            elif (use_2d == False) and (lateral == 'unilateral'):
                return BreastDIAGNOSISDatasetLR(**kwargs)
        elif cohort == 'subset':
            if (use_2d == True) and (lateral == 'bilateral'):
                return BreastDIAGNOSISSubsetDataset2D(**kwargs)
            elif (use_2d == True) and (lateral == 'unilateral'):
                return BreastDIAGNOSISSubsetDatasetLR2D(**kwargs)
            elif (use_2d == False) and (lateral == 'bilateral'):
                return BreastDIAGNOSISSubsetDataset(**kwargs)
            elif (use_2d == False) and (lateral == 'unilateral'):
                return BreastDIAGNOSISSubsetDatasetLR(**kwargs)
        else:
            raise Exception("Cohort name unknown.")
    else:
        raise Exception("Dataset name unknown.")

       

