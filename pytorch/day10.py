import pandas as pd
import numpy as np
import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score
from skorch import NeuralNetClassifier
torch.manual_seed(41)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
df=pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\employee_dataset_50000.csv")
#data handling and preporcessing