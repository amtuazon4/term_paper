import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

df = pd.read_csv("test_results11.csv", encoding="latin-1")
pred_val = df.iloc[:,0].values.tolist()
true_val = df.iloc[:,1].values.tolist()

result = f1_score(true_val, pred_val, average="weighted")
print(result)




