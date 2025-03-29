from src.preprocessing import *
from src import configs as conf

raw_data  = conf.RAW_DATA
n4_data   = conf.N4_DATA
crop_data = conf.CROPPED_DATA
pre_data  = conf.PRE_DATA
proc_data = conf.PROC_DATA
fold_data = conf.FOLD_DATA

if __name__ == "__main__":
    N4BiasFieldCorrection(raw_data, n4_data)
    Crop_3D_Images(n4_data, crop_data, size=192)
    Process_Images(crop_data, pre_data)
    Create_Dataset(pre_data, proc_data)
    Process_Folds(proc_data, fold_data, f=5)
    Struct_Folds(fold_data, fold_data)
