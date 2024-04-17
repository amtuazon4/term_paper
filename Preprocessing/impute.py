import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.impute import SimpleImputer

# This is imputes the data using mean imputation and exports it as a csv file
df = pd.read_csv("init_data.csv")
imp = SimpleImputer(missing_values=np.nan, strategy="mean")

vals = [list(x) for x in df.iloc[:,2:].values]
new_vals = [list(x) for x in imp.fit_transform(vals)]
df.iloc[:,2:] = new_vals

df.to_csv("init_data_imputed.csv", index=False)