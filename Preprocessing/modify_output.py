import pandas as pd
import numpy as np

# The following functions uses csv_file, and output_name parameters
# csv_file refers to the filename that will be read
# output_name refers to the output_name that will be used for the exported file

# Function for modifying the output variable class which is the variable that refers to the music genre
# uses binary representation
# also removes the artist name, and track name variable
def modify_output(csv_file, output_name):
    df = pd.read_csv(csv_file, encoding="latin-1")

    df.rename(columns={"Class":"Class1"}, inplace=True)
    class_ind = df.columns.get_loc("Class1")

    for i in range(1,4):
        df.insert(loc=class_ind+i, column=f"Class{1+i}", value=0.1)

    df["Class1"] = df["Class1"].astype(float)
    df["Class2"] = df["Class2"].astype(float)
    df["Class3"] = df["Class3"].astype(float)
    df["Class4"] = df["Class4"].astype(float)

    
    for i in range(df.shape[0]):
        genre = df.iloc[i,class_ind]
        bin_str = f"{int(genre):04b}".format()
        for ind , c in enumerate(bin_str):
            if(c=="0"):
                df.iloc[i, class_ind+ind] = 0.1
            else:
                df.iloc[i, class_ind+ind] = 0.9
    df.drop('Artist Name', axis=1, inplace=True)
    df.drop('Track Name', axis=1, inplace=True)
    
    df.to_csv(output_name, index=False)

# Function for modifying the output variable of the testing dataset
# Removes the Artist Name and Track Name variables
def modify_test_output(csv_file, output_name):
    df = pd.read_csv(csv_file, encoding="latin-1")
    
    df.drop('Artist Name', axis=1, inplace=True)
    df.drop('Track Name', axis=1, inplace=True)
    
    df.to_csv(output_name, index=False)

# Another function for modifying the output variable of a dataset which is Class
# This implemented by producing x number of output variables
# where x refers to the number of possible Classes or genres
# and each variable corresponds to a particular Class or genre
# For instance, if the output variable = 3, then the third output variable will have a 0l9
# while the rest will be 0.1
def modify_output2(csv_file, output_name):
    df = pd.read_csv(csv_file, encoding="latin-1")

    df.rename(columns={"Class":"Class1"}, inplace=True)
    class_ind = df.columns.get_loc("Class1")
    df["Class1"] = df["Class1"].astype(int)
    classes = df.iloc[:,class_ind].tolist()
    df["Class1"] = df["Class1"].astype(float)
    for i in range(1,11):
        df.insert(loc=class_ind+i, column=f"Class{1+i}", value=0.1)
        df[f"Class{1+i}"] = df[f"Class{1+i}"].astype(float)    

    df["Class1"] = 0.1 

    for i, num in enumerate(classes): df.iloc[i, class_ind+num] = 0.9

    df.drop('Artist Name', axis=1, inplace=True)
    df.drop('Track Name', axis=1, inplace=True)
    # print(df.iloc[:,18:])


    df.to_csv(output_name, index=False)


# Preprocesses the output variable of the testing, training, and validation datasets
modify_test_output("in_test_ds2.csv", "inout_test_ds2.csv")
modify_output("in_train_ds2.csv", "inout_train_ds2.csv")
modify_output("in_valid_ds2.csv", "inout_valid_ds2.csv")