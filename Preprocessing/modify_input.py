import pandas as pd
import numpy as np
import math

# The following functions uses csv_file, and output_name parameters
# csv_file refers to the filename that will be read
# output_name refers to the output_name that will be used for the exported file


# round the imputed class type data
def round_imputed_class_data(csv_file, output_name=None):
    df = pd.read_csv(csv_file, encoding="latin-1")
    classes = ["key", "mode", "time_signature"]
    for c in classes:
        col_num = df.columns.get_loc(c)
        for i, num in enumerate(df[c].tolist()):
            df.iloc[i,col_num] = round(num)
    df.to_csv(output_name, index=False)


# Function that preprocesses the input variables of the dataset
def modify_input(csv_file, output_name):
    df = pd.read_csv(csv_file, encoding="latin-1")
    
    # df["Popularity"] = df["Popularity"].astype(float)

    # to_proportional = ["Popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_in min/ms"]


    # for colname in to_proportional:
    #     col = df.columns.get_loc(colname)
    #     max_val = max(df.iloc[:,col])
    #     min_val = min(df.iloc[:,col])
    #     for row in range(len(df)):
    #         val = df.iloc[row, col]
    #         proportion = (val - min_val)/(max_val-min_val)
    #         df.iloc[row, col] = proportion*(0.8) + 0.1

    # preprocesses the key class variable using binary representation
    df.rename(columns={"key":"key1"}, inplace=True)
    df["key1"] = df["key1"].astype(float)
    df.insert(loc=df.columns.get_loc("key1")+1, column="key4", value=[0.1 for x in range(len(df))])
    df.insert(loc=df.columns.get_loc("key1")+1, column="key3", value=[0.1 for x in range(len(df))])
    df.insert(loc=df.columns.get_loc("key1")+1, column="key2", value=[0.1 for x in range(len(df))])
    
    for i, val in enumerate(df.iloc[:,5:9].values):
        temp = val[0]
        for j in range(8, 4, -1):
            if(temp % 2 == 1): df.iloc[i,j] = 0.9
            else:  df.iloc[i,j] = 0.1
            temp = float(temp//2)
    
    # preprocesses the mode class variable using binary representation
    mode_col = df.columns.get_loc("mode")
    df["mode"] = df["mode"].astype(float)
    for i in range(len(df)):
        if(df.iloc[i,mode_col] == 1): df.iloc[i,mode_col] = 0.9
        else: df.iloc[i,mode_col] = 0.1

    # preprocesses the time signature class variable using binary representation
    df.rename(columns={"time_signature":"time_signature1"}, inplace=True)
    df["time_signature1"] = df["time_signature1"].astype(float)
    df.insert(loc=df.columns.get_loc("time_signature1")+1, column="time_signature2", value=[0.1 for x in range(len(df))])

    ts_one = df.columns.get_loc("time_signature1")
    ts_two = ts_one+1

    for i, val in enumerate(df.iloc[:,ts_one:ts_two+1].values):
        if(val[0] == 1): df.iloc[i,ts_one:ts_two+1] = [0.1, 0.1]
        if(val[0] == 3): df.iloc[i,ts_one:ts_two+1] = [0.1, 0.9]
        if(val[0] == 4): df.iloc[i,ts_one:ts_two+1] = [0.9, 0.1]
        if(val[0] == 5): df.iloc[i,ts_one:ts_two+1] = [0.9, 0.9]

    df.to_csv(output_name, index=False)

# Function for interpolation
def interpolate(x0, x1, y0, y1, x):
    return y0 + (x - x0) * ((y1 - y0)/(x1 - x0))

# Function for preprocessing quantitative variables using interpolation representation
# the cols parameter determines the number of variables that will be used in the interpolation representation
# the inp parameter refers to the variable name that will be in the interpolation representation process
def ip_rpr(csv_file, output_name, inp, cols):
    df = pd.read_csv(csv_file, encoding="latin-1")
    
    for col in cols:
        max_val = max(df[col])
        min_val = min(df[col])
        ind = df.columns.get_loc(col)
        df.rename(columns={col:(col+"1")}, inplace=True)
        for i in range(inp, 1, -1):
            df.insert(loc=ind+1, column=(col+str(i)), value=[0.1 for x in range(len(df))])
        
        
        for i, val in enumerate(df.iloc[:, ind]):
            inter0 = interpolate(min_val, max_val, 0, inp-1, val)
            inter1 = interpolate(min_val, max_val, 1, inp, val)
            ref_val = math.ceil(inter0)
            df.iloc[i, ind] = 0.1
            if(ref_val - inter0 > inter1 - ref_val): 
                df.iloc[i, ind+(ref_val-1)] = 0.9
            else: df.iloc[i, ind+ref_val] = 0.9
        
    df.to_csv(output_name, index=False)


# Fixes the imputed class input variables of the imputed data by rouding the mean imputed values
# round_imputed_class_data("init_data_imputed.csv", "init_data_imputed2.csv")

# Preprocesses the class variables of the fixed imputed data
# modify_input("init_data_imputed2.csv", "input_mod2.csv")

# Preprocess the following quantitative variables of the training, validation, and testing datasets
quant_var = ["Popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_in min/ms"]
ip_rpr("inout_test_ds2.csv", "test5.csv", 4, quant_var)
ip_rpr("inout_train_ds2.csv", "train5.csv", 4, quant_var)
ip_rpr("inout_valid_ds2.csv", "valid5.csv", 4, quant_var)
