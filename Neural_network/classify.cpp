#include <iostream>
#include "nn.h"

int main(){
    // Create a Neural network from the data
    double initLR = 0.00001;
    Neural_net nn = import_weights("min_weight11.txt", &initLR);
    
    // Read the datasets
    nn.read_testing_data("test5.csv", 5399, 52);
    nn.read_training_data("train5.csv", 7198, 55, 51);
    nn.read_validation_data("valid5.csv", 5399, 55);
    
    // Classify the data in the testing dataset
    nn.generate_class();
    
    return 0;
}