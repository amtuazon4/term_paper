#include <iostream>
#include "nn.h"

int main(){
    // Creating a Neural Network
    Neural_net nn(51,1,576,4,0.00001);
    
    // Read the training, validation, and testing datasets
    nn.read_training_data("train5.csv", 7198, 55, 51);
    nn.read_validation_data("valid5.csv", 5399, 55);
    nn.read_testing_data("test5.csv", 5399, 52);
    
    // Parameters for Training
    double kappa = 0.01;
    double phi = 0.5;
    double theta = 0.7;
    double mu = 0.9;
    
    // For training the dataset
    nn.train(20000, &kappa, &phi, &theta, &mu);
    
    return 0;
}