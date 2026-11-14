
import os
import pandas as pd
from sys import path
from utils import ROOT


data_dir = os.path.join(ROOT, "data")
file = os.path.join(data_dir, "mini-llama-articles.csv")

mini_dataset = pd.read_csv(file)
print(mini_dataset.head())