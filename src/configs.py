# ===================================================================================
# MAIN PARAMETERS OF PROJECT (IDEALLY THIS SHOULD BE READ FROM A CONFIGURATION FILE)
# ===================================================================================

from pathlib import Path

PROJ_ROOT: Path = Path.cwd()

########################
# DIR TO DATASETS
#########################

RAW_DATA  = PROJ_ROOT / 'data/BraTS2019/'

# interim dataset for processing
INTERIM_DATA = PROJ_ROOT / 'data/interim/'
N4_DATA      = INTERIM_DATA / 'n4_data/'
CROPPED_DATA = INTERIM_DATA / 'cropped_data/'
PRE_DATA     = INTERIM_DATA / 'pre_data/' 
PROC_DATA    = INTERIM_DATA / 'proc_data/'
FOLD_DATA    = INTERIM_DATA / 'fold_data/'

train_data = PROJ_ROOT / 'data/processed/train/'
valid_data = PROJ_ROOT / 'data/processed/valid/'

Xtrain = train_data / 'Xtrain.npy'
Ytrain = train_data / 'Ytrain.npy'
Xval   = valid_data / 'Xval.npy'
Yval   = valid_data / "Yval.npy"

##################################################
# TAKE SAMPLE OF THE DATA IF YOU RAN OUT OF MEMORY
##################################################

SAMPLE = False
SAMPLE_RATIO = 0.5 #  0.5 means loading 50% of the data

#######################
# TRAINING PARAMETERS
#######################

BATCH_SIZE = 8
EPOCHS = 30

####################
# MODEL/RESULTS PATH
####################

models_path = PROJ_ROOT / 'models/'
results_path = PROJ_ROOT / "results"

models_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)