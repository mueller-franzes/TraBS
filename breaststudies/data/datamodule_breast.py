import torch 

from breaststudies.data import BaseDataModule,  BreastDataset, BreastDatasetLR, BreastDataset2D, BreastUKADatasetLR


class BreastDataModule(BaseDataModule):
    Dataset = BreastDataset
    label2rgb = torch.tensor([
                                [0,0,0],  # Background 
                                [255,0,0],   # Label 1 
                                [0, 255, 0], # Label 2 
                                [0, 0, 255], # Label 3
                                [255, 255, 0], # Label 4
                                [255, 0, 255], # Label 5
                                [0, 255, 255], # Label 6
                                [255, 255, 255] # Label 7
                            ], dtype=torch.uint8)


class BreastDataModuleLR(BreastDataModule):
    Dataset = BreastDatasetLR

class BreastDataModule2D(BreastDataModule):
    Dataset = BreastDataset2D

class BreastUKADataModuleLR(BreastDataModule):
    Dataset = BreastUKADatasetLR
    
   
 