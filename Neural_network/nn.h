#include <string>
#pragma once

// Header file for the Neural_net class
// Contains the attributes and the methods that will be used in the Neural_net object
class Neural_net {
    public:
        double u;
        double v;
        int numINodes;
        int numHLayer;
        int numHNodes;
        int numONodes;
        double initLR;
        double* in;
        double*** hid;
        double** out;
        double** y;
        double* z;
        double*** c_hid;
        double** c_out;
        double*** dhid;
        double** dout;
        double*** ehid;
        double** eout;
        double*** fhid;
        double** fout;
        double** q;
        double* p;
        double c;
        
        int out_ind;
        double** training_data;
        int train_r, train_c;

        double** validation_data;
        int valid_r, valid_c;

        double** testing_data;
        int test_r, test_c;
        
        Neural_net(int numINodes, int numHLayer, int numHNodes, int numONodes, double initLR);
        ~Neural_net();
        
        void init_weights();
        void init_outputs();
        void init_vars();
        void print_weights();
        void export_weights(int* e, int* m);
        void forward(double* input);
        
        void backward(double* true_values);
        void zero_pq();
        double sigmoid(double* value);
        void read_training_data(std::string filename, int num_row, int num_col, int output_index);
        void read_validation_data(std::string filename, int num_row, int num_col);
        void read_testing_data(std::string filename, int num_row, int num_col);
        void train(int e, double* kappa, double* phi, double* theta, double* momentum);
        double squared_error(double* true_values);
        void adj_weights(double* kappa, double* phi, double* theta, double* momentum);
        
        void init_test_weights();
        
        int classify(double* input);
        void generate_class();
        
};

// Header for the import_weights function
Neural_net import_weights(std::string name, double* i);