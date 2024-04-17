Name: Andre M. Tuazon
Student Number: 2020-00839

Files
1. init_data.csv
    - This is the dataset provided in the google classroom
    - renamed it to init_data.csv

2. impute.py
    - file for imputing the init_data.csv

3. modify_input.py
    - file for modifying the input variables of a dataset

4. modify_output.py
    - file for modifying the output variable of a dataset

5. divide_dataset.py
    - file for dividing the dataset using stratified splitting

6. 
in_test_ds2.csv
in_train_ds2.csv
in_valid_ds2.csv
init_data-imputed.csv
init_data-imputed2.csv
inout_test_ds2.csv
inout_train_ds2.csv
inout_valid_ds2.csv
    - modified datasets used during preprocessing

7. test5.csv
    - the final preprocessed dataset for the testing data

8. train5.csv
    - the final preprocessed dataset for the training data

9. valid5.csv
    - the final preprocessed dataset for the validation data

NOTE: Doing these procedures again will change the order of the data affect the final datasets

How was the data preprocessed
1. Impute the missing data values in init_data.csv using impute.py resulting to init_data_imputed.csv

2. Fix the imputed class data in modify_input.py resulting to init_data_imputed2.csv

3. Preprocess the class input variables of the fixed imputed data in modify_input.py resulting to input_mod2.csv

4. Divide the dataset into training, validation, and testing datasets using divide_dataset.py resulting to 
    - in_train_ds2.csv
    - in_valid_ds2.csv
    - in_test_ds2.csv

5. Preprocess the output variable of the datasets using modify_output.py resulting to 
    - inout_test_ds2.csv
    - inout_train_ds2.csv
    - inout_valid_ds2.csv

6. Preprocess the quantitative input variables of the datasets in modify_input.py resulting to
    - test5.csv
    - train5.csv
    - valid5.csv