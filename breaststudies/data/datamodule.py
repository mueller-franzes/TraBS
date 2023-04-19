from pathlib import Path
import yaml
import itertools
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler, WeightedRandomSampler
import torch.multiprocessing as mp 




class BaseDataModule(pl.LightningDataModule):

    Dataset = torch.utils.data.Dataset

    def __init__(self,
                 batch_size: int = 1,
                 val_set: bool = True,
                 test_set: bool = True, 
                 n_splits=5,
                 num_workers: int = mp.cpu_count(),
                 seed: int = 0, 
                 path_root: str = None,
                 **ds_kwargs):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')
        self.hyperparameters.update(**ds_kwargs)
        self.hyperparameters.pop('ds_kwargs')

        self.batch_size = batch_size
        self.val_set = val_set
        self.test_set = test_set
        self.n_splits = n_splits
        self.num_workers = num_workers
        self.seed = seed 

        self.path_root = self.Dataset.path_root_default if path_root is None else path_root
        self.ds_kwargs = ds_kwargs

        self._item_pointers_split = {} 


    def save(self, path_config_dir, config_file='data_conf.yaml', split_file='data_split.yaml'):
        path_config_dir = Path(path_config_dir)
        path_config_dir.mkdir(parents=True, exist_ok=True)
        # Get parameters set during 'setup()' or get default values 
        for ds_name in ['ds_train', 'ds_val', 'ds_test']:
            ds = getattr(self, ds_name, None)
            if ds is not None:
                params = dict(ds.hyperparameters)
                params.pop('path_root')
                params.pop('item_pointers')
                self.hyperparameters['params_'+ds_name] = params
        with open(path_config_dir / config_file, "w", newline="") as fp:
            yaml.dump(self.hyperparameters, fp)
        with open(path_config_dir / split_file, "w", newline="") as fp:
            yaml.dump(self._item_pointers_split, fp)


    @classmethod
    def load(cls, path_config_dir, config_file='data_conf.yaml', split_file='data_split.yaml', **kwargs):
        with open(Path(path_config_dir) / config_file, 'r') as f:
            hyperparameters = yaml.load(f, Loader=yaml.UnsafeLoader)
        hyperparameters.update(kwargs)
        dm = cls(**hyperparameters)
        dm._item_pointers_split = cls.load_split(path_config_dir, split_file)
        return dm 

    
    @classmethod
    def load_split(self, path_config_dir, split_file='data_split.yaml'):
        path_split_file = Path(path_config_dir) / split_file
        if path_split_file.is_file():
            with open(Path(path_config_dir) / split_file, 'r') as f:
                split_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        else:
            raise Exception('File not found: {}'.format(path_split_file))
        return split_dict
    
    

    def setup(self, stage=None, split=0, params_ds_train={}, params_ds_val={}, params_ds_test={} ):
        # Create new split if not already exists
        if len(self._item_pointers_split) ==0:
            self._item_pointers_split = self.Dataset.groupkfold(self.path_root, self.n_splits, self.val_set, self.test_set)
     
        self.setup_split(stage=stage, split=split, params_ds_train=params_ds_train, params_ds_val=params_ds_val, params_ds_test=params_ds_test)

    def setup_split(self, stage=None, split=0, params_ds_train={}, params_ds_val={}, params_ds_test={}):
        item_pointers_split = self._item_pointers_split[split]
        val_set = 'val' in item_pointers_split
        test_set = 'test' in item_pointers_split

        ds_params = self.ds_kwargs
        if stage == 'fit' or stage is None:
            params =  {} 
            params.update({k: ds_params[k] for k in ds_params.keys() - {'params_ds_train', 'params_ds_val', 'params_ds_test'}})  
            params.update(ds_params.get('params_ds_train', {})) # Overwrite with dataset specific parameters 
            params.update(params_ds_train) # Overwrite with dataset specific parameters passed to setup() call 
            params['path_root'] = self.path_root
            params['item_pointers'] = item_pointers_split['train']
            self.ds_train = self.Dataset(**params)

            if val_set:
                params =  {} 
                params.update({k: ds_params[k] for k in ds_params.keys() - {'params_ds_train', 'params_ds_val', 'params_ds_test'}})  
                params.update(ds_params.get('params_ds_val', {}))
                params.update(params_ds_val) # FIXME: Add function for nested dict update 
                params['path_root'] = self.path_root
                params['item_pointers'] = item_pointers_split['val']
                self.ds_val = self.Dataset(**params)

        if stage == 'test' or stage is None:
            if not test_set:
                raise AssertionError("A test test set was not requested during initialization. Adjust your settings.")
            params =  {} 
            params.update({k: ds_params[k] for k in ds_params.keys() - {'params_ds_train', 'params_ds_val', 'params_ds_test'}}) 
            params.update(ds_params.get('params_ds_test', {}))
            params.update(params_ds_test)
            params['path_root'] = self.path_root
            params['item_pointers'] = item_pointers_split['test']
            self.ds_test = self.Dataset(**params)

    

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # NOTE: Speed up for 2D: Load (in random order) all slices within one 3D volume before proceeding with next volume 
        # return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, 
        #                   sampler=self.rand_series_sampler(self.ds_train)) 
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=RandomSampler(self.ds_train, replacement=True, num_samples=len(self.ds_train)), # NOTE: nnUNet default is num_samples=250
                            generator=generator,
                          drop_last=True)

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.val_set:
            return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, 
                                generator=generator
                              )
        else:
            raise AssertionError("A validation set was not requested during initialization. Adjust your settings.")

    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.test_set:
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, 
                            #   worker_init_fn=self.worker_init_fn
                            generator = generator,
                              )
        else:
            raise AssertionError("A test test set was not requested during initialization. Adjust your settings.")

    # def worker_init_fn(self, worker_id):
    #     # Note: By default, all workers have different seeds (important for data augmentation)
    #     # worker_info = torch.utils.data.get_worker_info()
    #     if self.seed is not None: 
    #         # np.random.seed(self.seed) # may be overwritten or ignored by other modules 
    #         # np.random.default_rng(self.seed) 
    #         torch.manual_seed(self.seed)


    @classmethod
    def rand_series_sampler(cls, dataset, generator=None):
        class SeriesRandomSampler(RandomSampler):
            """Items are sampled randomly but all items within a series are subsequently (randomly) sampled."""
            def __iter__(self):
                # Get number of series and number of items per series 
                series_pointers = self.data_source.get_series_pointers() 
                n_series = len(series_pointers)
                n_items = [len(series_pointer) for series_pointer in series_pointers.values() ]


                # Create list of indices per series element 
                counter = itertools.count(0)
                idxs = [[next(counter) for _ in item_pointers] for item_pointers in series_pointers.values()]

                if self.generator is None:
                    generator = torch.Generator()
                    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                else:
                    generator = self.generator

                if self.replacement:
                    raise NotImplementedError()
                else:
                    for series_idx in torch.randperm(n_series, generator=self.generator):
                        for sires_item_idx in torch.randperm(n_items[series_idx], generator=self.generator):
                            yield idxs[series_idx][sires_item_idx]
        
        return SeriesRandomSampler(dataset, generator=generator)


    @classmethod
    def weighted_series_sampler(cls, dataset, model, generator=None):
        class SeriesWeightedRandomSampler(WeightedRandomSampler):
            def __init__(self, dataset, weights, num_samples: int, replacement: bool = True, generator=None) -> None:
                if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
                    raise ValueError("num_samples should be a positive integer "
                                    "value, but got num_samples={}".format(num_samples))
                if not isinstance(replacement, bool):
                    raise ValueError("replacement should be a boolean value, but got "
                                    "replacement={}".format(replacement))
                self.weights = weights
                self.num_samples = num_samples
                self.replacement = replacement
                self.generator = generator
                self.data_source = dataset 

            def __iter__(self):
                # Get number of series and number of items per series 
                series_pointers = self.data_source.get_series_pointers() 
                n_series = len(series_pointers)
                n_items = [len(series_pointer) for series_pointer in series_pointers.values() ]

                # Create list of indices per series element 
                counter = itertools.count(0)
                idxs = [[next(counter) for _ in item_pointers] for item_pointers in series_pointers.values()]
    

                for series_idx in torch.multinomial(torch.as_tensor(self.weights[0], dtype=torch.double), n_series, self.replacement, generator=self.generator):
                    for sires_item_idx in torch.multinomial(torch.as_tensor(self.weights[1][series_idx], dtype=torch.double), n_items[series_idx], self.replacement, generator=self.generator):
                        yield idxs[series_idx][sires_item_idx]



        # -------------- Compute weights --------------
        sample_weights = [[], []] # [[series1, series2, ...], [[item1,item2], [item1, item2], ...]]
        for _, item_pointers in tqdm(dataset.get_series_pointers().items()):
            sample_weights[1].append([])
            for item_pointer in item_pointers:
                item = dataset.load_item(item_pointer)
                target = item['target']
                weight = sum([label_val in target for label_val in dataset.labels.values()])
                sample_weights[1][-1].append(weight) 
            sample_weights[0].append(max(sample_weights[1][-1]))


        return SeriesWeightedRandomSampler(dataset, weights=sample_weights, num_samples=len(dataset), replacement=True, generator=generator)