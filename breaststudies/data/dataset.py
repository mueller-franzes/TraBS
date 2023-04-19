from pathlib import Path
from xml.etree.ElementInclude import include

import numpy as np 
from sklearn.model_selection import GroupKFold, ShuffleSplit, StratifiedGroupKFold
from sklearn.utils import shuffle


class BaseDataset():
    path_root_default = Path('') 

    default_series_trans = None 
    default_item_trans = None 
    
    def __init__(self, path_root=None, item_pointers=[], cache=True, ram=False, series_trans=True, item_trans=True, **kwargs):
        """Load data from a directory and return standard format: image, mask, (patient)id
        
        Keyword Arguments:
            path_root {str, Path} -- Path to root directory (default {None = path_root_default})
            item_pointers {list} -- List of unique pointers (e.g. file paths) to all items. 
            ram {bool} -- Load entire data into RAM. Transformations are only applied when loading (not stored).
            cache {bool}, -- Load (Temprally) all items of a series into the ram. 
            series_trans {object} -- Composition of transformations applied to a series of connected items (e.g. 3D volume).  
            item_trans {object} -- Composition of transformations applied to the a single item (e.g. 2D patch).   
        """
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('self')
        self.hyperparameters.update(**kwargs)
        self.hyperparameters.pop('kwargs')

        self.path_root = self.path_root_default if path_root is None else Path(path_root)
        
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.get_item_pointers(self.path_root, **kwargs) 
        
        if ram:
            self.data = {item_pointer: self.load_series(item_pointer, self.path_root, **self.kwargs) 
                         for item_pointer in self.item_pointers}
        else:
            self.data = {} 

        
        self.item_iter = iter(self.item_pointers)
        self.ram = ram 
        self.series_trans = (self.default_series_trans if series_trans else None)  if isinstance(series_trans, bool) else series_trans  
        self.item_trans = (self.default_item_trans if item_trans else None)  if isinstance(item_trans, bool) else item_trans
        self.kwargs = kwargs
        self.cache = cache 
        self.cached_items = {}

        self._post_init()

    def __len__(self):
        return len(self.item_pointers)

    def __iter__(self):
        self.item_iter = iter(self.item_pointers) 
        return self
    
    def __next__(self):
        item_pointer = next(self.item_iter)
        data = self.load_item(item_pointer)
        return data 
    
    def __getitem__(self, idx):
        item_pointer = self.item_pointers[idx]
        data = self.load_item(item_pointer)
        return data 
    
    def load_item(self, item_pointer):
        item_pointer = tuple(item_pointer)
        if self.cache and item_pointer in self.cached_items:
            # item = self.cached_items.pop(item_pointer)
            item = self.cached_items.get(item_pointer)
        else:
            if self.ram: 
                series = self.data[item_pointer] 
            else:
                series = self.load_series(item_pointer, self.path_root, **self.kwargs)
            series = self.apply_transformation(series, self.series_trans)
            series_items = self.series2items(item_pointer[:max(1, len(item_pointer)-1)], series, **self.kwargs)

            item = series_items.get(item_pointer, None)
            assert item is not None, f"Could not find item_pointer: {item_pointer}"
            self.cached_items = series_items if self.cache else None 
        item = self.apply_transformation(item, self.item_trans)
        item = self.item2out(item, **self.kwargs)
        return item 
    
       
    def get_item_from_series(self, series, item_pointer, series_trans=True, item_trans=True, **kwargs):
        kwargs = {**self.kwargs, **kwargs} # replace values from self.kwargs if value exits in kwargs
        series_trans = (self.series_trans if series_trans else None)  if isinstance(series_trans, bool) else series_trans 
        item_trans = (self.item_trans if item_trans else None)  if isinstance(item_trans, bool) else item_trans 
        series = self.apply_transformation(series, series_trans)
        series_items = self.series2items(item_pointer[:max(1, len(item_pointer)-1)], series, **kwargs)
        item = series_items.get(item_pointer)
        item = self.apply_transformation(item, item_trans)
        item = self.item2out(item, **kwargs)
        return item 

    def get_labels(self, inc_label_names=[], exc_label_names=[]):
        inc_label_names = [inc_label_names] if isinstance(inc_label_names, str) else inc_label_names
        exc_label_names = [exc_label_names] if isinstance(exc_label_names, str) else exc_label_names
        if len(inc_label_names) == 0:
            inc_label_names = self.labels.keys()
        label_names = set(inc_label_names)-set(exc_label_names)        
        return {label_name: self.labels[label_name] for label_name in label_names }
    
    def get_label_fcts(self, inc_label_names=[], exc_label_names=[]):
        inc_label_names = [inc_label_names] if isinstance(inc_label_names, str) else inc_label_names
        exc_label_names = [exc_label_names] if isinstance(exc_label_names, str) else exc_label_names
        if len(inc_label_names) == 0:
            inc_label_names = self.label_fcts.keys()
        label_names = set(inc_label_names)-set(exc_label_names)        
        return {label_name: self.label_fcts[label_name] for label_name in label_names }

    @classmethod
    def get_item_pointers(cls, path_root=None, **kwargs):
        path_root = cls.path_root_default if path_root is None else Path(path_root)
        return [item for item in cls.init_crawler(path_root, **kwargs)]
    
    def get_series_pointers(self):
        series_pointers = {}
        for item_pointer in self.item_pointers:
            series_pointer = self.item_pointer2uid(item_pointer, path_root=self.path_root, id_type='series')
            if series_pointer in series_pointers:
                series_pointers[series_pointer].append(item_pointer)
            else:
                series_pointers[series_pointer] = [item_pointer]
        return series_pointers

    @classmethod
    def groupkfold(cls, path_root=None, n_splits=5, val_split=True, test_split=True, y=None, **kwargs):
        path_root = cls.path_root_default if path_root is None else Path(path_root)
        item_pointers = np.asarray(cls.get_item_pointers(path_root, **kwargs))      

        # Shuffle pointers before splitting them 
        item_pointers = shuffle(item_pointers, random_state=0)         
      
        pat_ids = np.asarray([cls.item_pointer2uid(item_pointer, path_root, id_type='patient') 
                               for item_pointer in item_pointers])   
        splitter = GroupKFold(n_splits=n_splits) # WANRING: Independent of seed state 
        # splitter = StratifiedGroupKFold(n_splits=n_splits)
         

        splits = []
        for idx_train, idx_test in splitter.split(pat_ids, groups=pat_ids):
            if val_split:
                if test_split:
                    # NOTE: Don't do inner/nested cross validation
                    subidx_train, subidx_val = list(splitter.split(idx_train, groups=pat_ids[idx_train]))[0]
                    splits.append({'train': item_pointers[idx_train[subidx_train]].tolist(), 
                            'val':  item_pointers[idx_train[subidx_val]].tolist(), 
                            'test': item_pointers[idx_test].tolist()})
                else:
                    splits.append({'train': item_pointers[idx_train].tolist(), 'val': item_pointers[idx_test].tolist()})
            else:
                splits.append({'train':  item_pointers[idx_train].tolist(), 'test': item_pointers[idx_test].tolist()})
        return splits 

    @classmethod
    def apply_transformation(cls, data, transformation=None):
        try:
            return data if transformation is None else transformation(data)
        except Exception as e:
            print("Failed to apply transformation: ", e)
            # return data

    

   
    # -------------------------------- Overwrite necessary functions---------------------------
    @classmethod
    def init_crawler(cls, path_root=None, **kwargs):
        """Return a iteratable pointer over all items e.g. path_root.rglob('*.jpg')"""
        raise NotImplementedError

    @classmethod
    def item_pointer2uid(cls, item_pointer, path_root=None, id_type='item'): 
        """Returns identifier for each item.

        Args:
            item_pointer (object): Unique pointer (e.g. Path or (Path, Slice)) to each item 
            path_root (Path, optional): Path to root directory 
            id_type (str, optional): Specifies the id-type 
        """
        if id_type=="patient": # A patient-ID can include several studies (req. for GroupKFold)
            raise NotImplementedError
        elif id_type == "study":  # A study-ID can include several s-IDs
            raise NotImplementedError
        elif id_type == "series": # A series-ID can include several item-IDs
            raise NotImplementedError
        elif id_type == "item": # An item (image/slice) -ID  is allways unique
            raise NotImplementedError

    @classmethod
    def load_series(cls, item_pointer, path_root=None, **kwargs):
        "Load data and return as dict e.g. {'uid':uid, 'source':source, 'target':target}"
        raise NotImplementedError  

    # -------------------------------- Overwrite usefull functions---------------------------
    def _post_init(self):
        pass 

    @classmethod
    def series2items(cls, item_pointer, series, **kwargs):
        return {item_pointer: series}
    
    @classmethod
    def item2out(cls, item, **kwargs):
        return item 
           
    @property
    def labels(self):
        """Return a dict of label names and their label values e.g. {'Lesion A':1, 'Lesion B':2} 
        
        Note:
            get_labels()
        """
        raise NotImplementedError
    
    @property
    def label_fcts(self):
        """Return a dict of label names (and combinations) and their mask function e.g. {'All Lesions':lambda x:x>=1}. 
        
        Note:
            get_label_fcts()
        """
        raise NotImplementedError
