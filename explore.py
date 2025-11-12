import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import gc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from IPython.display import display
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True, precision=3)
pd.set_option("display.float_format", "{:.3f}".format)

device = 'cpu'
try:
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
except:
    pass
print(f"Running models on device: {device}")

ROOT = "Isabella-ML"
DATASET = "High_quality_2019.csv"
PATH = os.path.join(os.getcwd(), ROOT, DATASET)
print(os.listdir(os.path.join(os.getcwd(), ROOT)))
df = pd.read_csv(PATH)
print(df)

SEED = 42
FOLDER = "Isabella-ML"
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

print(reduce_mem_usage(df))

variables_df = (
    pd.read_csv(f'{FOLDER}/{DATASET}', low_memory=True)
    .sort_values(by=['eom'])
    .reset_index(drop=True)
)
print(variables_df.memory_usage(deep=True).sum())

print(variables_df.head())

print(variables_df.isnull().sum())

## plots

variables_df["debt_me"].hist(bins=30) # Histograma
plt.title(f"Histogram of {"ret_exc_lead1m"}")
plt.show()