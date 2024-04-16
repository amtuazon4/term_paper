#include "nn.h"
#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <sstream>
#include <utility>
#include <cmath>
#include <iomanip>

// Constructor for the Neural_net class
Neural_net:: Neural_net(int inp, int l, int n, int o, double e){
    numINodes = inp;
    numHLayer = l;
    numHNodes = n;
    numONodes = o;
    initLR = e;
    
    init_weights();     // initialize weights
    init_outputs();     // initialize arrays for y and z
    init_vars();        //initialize learning rate, f, and d
};

// Destructor for the Neural_net class
Neural_net::~Neural_net(){
    int i, j, k;
    for(i=0; i< numHLayer; i++){
        for(int j=0; j< numHNodes; j++){
            delete[] hid[i][j];
            delete[] c_hid[i][j];
            delete[] dhid[i][j];
            delete[] ehid[i][j];
            delete[] fhid[i][j];
        }
        delete[] hid[i];
        delete[] c_hid[i];
        delete[] dhid[i];
        delete[] ehid[i];
        delete[] fhid[i];
        delete[] q[i];
    }
    delete[] hid;
    delete[] c_hid;
    delete[] dhid;
    delete[] ehid;
    delete[] fhid;
    delete[] q;

    for(i=0; i< numONodes; i++){
        delete[] out[i];
        delete[] c_out[i];
        delete[] dout[i];
        delete[] eout[i];
        delete[] fout[i];
    }
    delete[] out;
    delete[] c_out;
    delete[] dout;
    delete[] eout;
    delete[] fout;
    delete[] p;
    

    for(i=0; i<train_r; i++){
        delete[] training_data[i];
    }
    delete[] training_data;

    for(i=0; i<valid_r; i++){
        delete[] validation_data[i];
    }
    delete[] validation_data;

    for(i=0; i<test_r; i++){
        delete[] testing_data[i];
    }
    delete[] testing_data;
    


    std::cout << "Destructing complete\n";
}

// Method for initializing the weights of the neural network
void Neural_net::init_weights(){
    int i, j, k;
    hid = new double**[numHLayer];
    for (i=0; i<numHLayer; i++){
        hid[i] = new double*[numHNodes];
        if(i==0){
            for (j=0; j<numHNodes; j++) hid[i][j] = new double[numINodes+1];
        }else{
            for (j=0; j<numHNodes; j++) hid[i][j] = new double[numHNodes+1];
        }
    }

    out = new double*[numONodes];
    for (i=0; i<numONodes; i++) out[i] = new double[numHNodes+1];
    
    // Randomizes hidden weights using numbers between 0.0 and 0.1 
    // Also uses a normal distribution in selecting the numbers
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 0.1);
    
    for(i=0; i<numHLayer; i++){
        for(j=0; j<numHNodes; j++){
            if(i==0){
                for(k=0; k<numINodes+1; k++) 
                    hid[i][j][k] = distribution(generator);
            }else {
                for(k=0; k<numHNodes+1; k++) 
                    hid[i][j][k] = distribution(generator);
            }
        }
    }            

    // If the total number of weights involved in an output node is even,
    // then half of the weights are assigned with 1 and other half with -1
    if((numHNodes+1) % 2 == 0){
        double* temp = new double[numHNodes+1];
        for(i=0; i<numONodes; i++){
            for(j=0; j<numHNodes+1; j++) temp[j] = (j%2==0) ? 1 : -1;
            std::shuffle(temp, temp+(numHNodes+1), generator);
            for(j=0; j<numHNodes+1; j++) out[i][j] = temp[j];
        }
        delete[] temp;
    
    // Otherwise, the bias weight will be assigned with 0
    // Meanwhile, half of the hidden weights are assigned with 1 and other half with 0
    }else{
        double* temp = new double[numHNodes];
        for(i=0; i<numONodes; i++){
            out[i][0] = 0;
            for(j=0; j<numHNodes; j++) temp[j] = (j%2==0) ? 1 : -1;
            std::shuffle(temp, temp+numHNodes, generator);
            for(j=1; j<numHNodes+1; j++) out[i][j] = temp[j-1];
        }
        delete[] temp;
    }
};

// Method for initializing the y and z arrays
void Neural_net::init_outputs(){
    int i, j, k;
    y = new double*[numHLayer];
    for (i=0; i<numHLayer; i++) y[i] = new double[numHNodes];
    z = new double[numONodes];
}

// Initiate the variables for momentum and adaptive learning rates
void Neural_net::init_vars(){
    int i, j, k;
    c_hid = new double**[numHLayer];
    dhid = new double**[numHLayer];
    ehid = new double**[numHLayer];
    fhid = new double**[numHLayer];
    q = new double*[numHLayer];
    

    for (i=0; i<numHLayer; i++){
        c_hid[i] = new double*[numHNodes];
        dhid[i] = new double*[numHNodes];
        ehid[i] = new double*[numHNodes];
        fhid[i] = new double*[numHNodes];
        q[i] = new double[numHNodes];
        if(i==0){
            for (j=0; j<numHNodes; j++) {
                c_hid[i][j] = new double[numINodes+1];
                dhid[i][j] = new double[numINodes+1];
                ehid[i][j] = new double[numINodes+1];
                fhid[i][j] = new double[numINodes+1];
            }
        }else{
            for (j=0; j<numHNodes; j++) {
                c_hid[i][j] = new double[numHNodes+1];
                dhid[i][j] = new double[numHNodes+1];
                ehid[i][j] = new double[numHNodes+1];
                fhid[i][j] = new double[numHNodes+1];
            }
        }
    }
    c_out = new double*[numONodes];
    dout = new double*[numONodes];
    eout = new double*[numONodes];
    fout = new double*[numONodes];
    p = new double[numONodes];
    for (i=0; i<numONodes; i++){
        c_out[i] = new double[numHNodes+1];
        dout[i] = new double[numHNodes+1];
        eout[i] = new double[numHNodes+1];
        fout[i] = new double[numHNodes+1];
    }

    for(i=0; i<numHLayer; i++){
        for(j=0; j<numHNodes; j++){
            if(i==0){
                for(k=0; k<numINodes+1; k++){
                    c_hid[i][j][k] = 0;
                    dhid[i][j][k] = 0;
                    ehid[i][j][k] = initLR;
                    fhid[i][j][k] = 0;
                } 
            }else {
                for(k=0; k<numHNodes+1; k++) {
                    c_hid[i][j][k] = 0;
                    dhid[i][j][k] = 0;
                    ehid[i][j][k] = initLR;
                    fhid[i][j][k] = 0;
                } 
            }
        }
    } 

    for (i=0; i<numONodes; i++){
        for(j=0; j<numHNodes+1; j++){
            c_out[i][j] = 0;
            dout[i][j] = 0;
            eout[i][j] = initLR;
            fout[i][j] = 0;
        }
    }
    
}

// Method for assigning zero to all the elements of arrays p and q
void Neural_net::zero_pq(){
    int i, j;
    for(i=0; i<numONodes; i++) p[i] = 0;
    
    for(i=0; i<numHLayer; i++) {
        for(j=0; j<numHNodes; j++) q[i][j]=0;
    }
}

// Method for printing the weights
void Neural_net::print_weights(){
    int i, j, k;
    std::cout << numINodes << " " << numHLayer << " " << numHNodes << " " << numONodes << " " << std::endl;
    double*** temp = hid;
    for(i=0; i<numHLayer; i++){
        for(j=0; j<numHNodes; j++){
            if(i==0){
                for(k=0; k<numINodes+1; k++) {
                    std::cout << temp[i][j][k] << " ";
                }
            }else{
                for(k=0; k<numHNodes+1; k++) {
                    std::cout << temp[i][j][k] << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    double** temp2 = out;
    for(i=0; i<numONodes; i++){
        for(j=0; j<numHNodes+1; j++){
            std::cout << temp2[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Method for exporting the weights of every epoch to the weights.txt file
// Also exports the weight with the minimum validation error to min_weight.txt
void Neural_net::export_weights(int* epoch, int* mode){
    int i, j, k;
    std::ofstream fp;
    if(*mode == 0) fp.open("weights.txt", std::ios::app);
    else fp.open("min_weight.txt");
    fp << "Epoch: " << *epoch << std::endl;
    fp << numINodes << " " << numHLayer << " " << numHNodes << " " << numONodes << " " << initLR << std::endl;
    for(i=0; i<numHLayer; i++){
        for(j=0; j<numHNodes; j++){
            fp << i << " " << j << " ";
            if(i==0){
                for(k=0; k<numINodes+1; k++) {
                    fp << hid[i][j][k] << " ";
                }
            }else{
                for(k=0; k<numHNodes+1; k++) {
                    fp << hid[i][j][k] << " ";
                }
            }
            fp << std::endl;
        }
    }

    for(i=0; i<numONodes; i++){
        fp << i << " ";
        for(j=0; j<numHNodes+1; j++){
            fp << out[i][j] << " ";
        }
        fp << "\n";
    }
    fp << "\n";
    fp.close();
}

// Initialize a Neural_net object based on an exported weight file
// Returns a Neural_net object
// The initLR parameter refers to the initial learning rate that the Neural_net object should have
Neural_net import_weights(std::string filename, double* initLR){
    int params[4], i, j, layer, node;
    std::string line, token;
    std::ifstream fp(filename);
    
    std::getline(fp, line);
    std::getline(fp, line);
    std::stringstream ss(line);
    for(i=0; i<4; i++){
        ss >> token;
        params[i] = std::stoi(token);
    }
    // ss >> token;
    // initLR = std::stod(token);

    Neural_net nn(params[0], params[1], params[2], params[3], *initLR);

    for(i=0; i<(nn.numHLayer * nn.numHNodes); i++){
        std::getline(fp, line);
        std::stringstream ss(line);
        j = 0;
        ss >> token;
        layer = std::stoi(token);
        ss >> token;
        node = std::stoi(token);
        while(ss >> token){
            nn.hid[layer][node][j++] = stod(token);
        }
    }

    for(i=0; i<(nn.numONodes); i++){
        std::getline(fp, line);
        std::stringstream ss(line);
        j = 0;
        ss >> token;
        node = std::stoi(token);
        while(ss >> token){
            nn.out[node][j++] = stod(token);
        }
    }

    fp.close();

    return nn;
}

// Method for the sigmoid function
double Neural_net::sigmoid(double* value){
    if(*value < -88) return 0;
    if(*value > 88) return 1;
    return 1 / (1 + 1/std::exp(*value));
}

// Method for the forward propagation
// The inp parameter accepts an array of inputs
void Neural_net::forward(double* inp){
    int i, j, k;
    in = inp;
    for(i=0; i<numHLayer; i++){
        if(i==0){
            for(j=0; j<numHNodes; j++){
                u = hid[i][j][0];
                for(k=0; k<numINodes; k++) u += inp[k] * hid[i][j][k+1];
                y[i][j] = sigmoid(&u);
            }
        }else{
            for(j=0; j<numHNodes; j++){
                u = hid[i][j][0];
                for(k=0; k<numHNodes; k++) u += y[i-1][k] * hid[i][j][k+1];
                y[i][j] = sigmoid(&u);
            }
        }
    }
    for(i=0; i<numONodes; i++){
        v = out[i][0];
        for(j=0; j<numHNodes; j++) v += y[numHLayer-1][j] * out[i][j+1];
        z[i] = sigmoid(&v);
    }
}

// Method for the backpropagation
// Accepts the t parameter which should be an array of expected values that the output layer should have
void Neural_net::backward(double* t){
    int i, j, k;
    zero_pq();
    for(i=0; i<numONodes; i++){
        p[i] = (z[i] - t[i]) * z[i] * (1 - z[i]);
        dout[i][0] += p[i];
        for(j=0; j<numHNodes; j++){
            dout[i][j+1] += p[i] * y[numHLayer-1][j];
            q[numHLayer-1][j] += p[i] * out[i][j+1];
        }
    }

    for(i=(numHLayer-1); i>-1; i--){
        if(i!=0){
            for(j=0; j<numHNodes; j++){
                q[i][j] *= y[i][j] * (1 - y[i][j]);
                dhid[i][j][0] += q[i][j];
                for(k=0; k<numHNodes; k++){
                    dhid[i][j][k+1] += q[i][j] * y[i-1][i];
                    q[i-1][j] += q[i][j] * hid[i][j][i+1];
                }
            }
        }else{
            for(j=0; j<numHNodes; j++){
                q[i][j] *= y[i][j] * (1 - y[i][j]);
                dhid[i][j][0] += q[i][j];
                for(k=0; k<numINodes; k++){
                    dhid[i][j][k+1] += q[i][j] * in[k];
                }
            }
        }
    }
}

// Method to read the training data
// The f parameter refers to the filename assumming its a csv file
// The r parameter refers to the number of rows the file has
// The c parameter refers to the number of columns the file has
// The o parameter refers to the index where the output values start

void Neural_net::read_training_data(std::string f, int r, int c, int o){
    int i, j, k;
    std::ifstream fp(f);
    std::string line, token;
    std::getline(fp, line);
    out_ind = o;
    train_r = r;
    train_c = c;
    training_data = new double*[r];
    for(i=0; i<r; i++){
        training_data[i] = new double[c];
    }
    
    i=0;
    while(std::getline(fp, line)){
        std::stringstream ss(line);
        for(j=0; j<c; j++){
            std::getline(ss, token, ',');
            training_data[i][j] = std::stod(token);
        }
        i++;
    }
}

// Method to read the validation data
// The f parameter refers to the filename assumming its a csv file
// The r parameter refers to the number of rows the file has
// The c parameter refers to the number of columns the file has
void Neural_net::read_validation_data(std::string f, int r, int c){
    int i, j, k;
    std::ifstream fp(f);
    std::string line, token;
    std::getline(fp, line);
    valid_r = r;
    valid_c = c;
    validation_data = new double*[r];
    for(i=0; i<r; i++){
        validation_data[i] = new double[c];
    }
    
    i=0;
    while(std::getline(fp, line)){
        std::stringstream ss(line);
        for(j=0; j<c; j++){
            std::getline(ss, token, ',');
            validation_data[i][j] = std::stod(token);
        }
        i++;
    }
}

// Method to read the testing data
// The f parameter refers to the filename assumming its a csv file
// The r parameter refers to the number of rows the file has
// The c parameter refers to the number of columns the file has
void Neural_net::read_testing_data(std::string f, int r, int c){
    int i, j, k;
    std::ifstream fp(f);
    std::string line, token;
    std::getline(fp, line);
    test_r = r;
    test_c = c;
    testing_data = new double*[r];
    for(i=0; i<r; i++){
        testing_data[i] = new double[c];
    }
    i=0;
    while(std::getline(fp, line)){
        std::stringstream ss(line);
        for(j=0; j<c; j++){
            std::getline(ss, token, ',');
            testing_data[i][j] = std::stod(token);
        }
        i++;
    }
}

// Method for training the Neural network
// The epoch parameter refers to the number epochs that the neural network will train
// The ka parameter refers to the kappa value
// The p parameter refers to the phi value
// The t parameter refers to the theta value
// The m parameter refers to the mu value
void Neural_net::train(int epoch, double* ka, double* p, double* t, double* m){
    int i, j, k;
    int mode = 0;
    double training_error, validation_error, min;
    std::ofstream fp; 
    fp.open("weights.txt");
    fp.close();
    fp.open("results.csv");
    fp.close();
    std::cout << std::fixed;
    std::cout << std::setprecision(8);
    fp.open("results.csv");
    
    fp << numINodes << ","
    << numHLayer << ","
    << numHNodes << ","
    << numONodes << ","
    << initLR << ","
    << epoch << ","
    << *ka << ","
    << *p << ","
    << *t << ","
    << *m << " " << "mean imputated o" << std::endl;

    
    for(i=0; i<epoch; i++){
        training_error = validation_error = 0;
        for(j=0; j<train_r; j++){
            forward(training_data[j]);
            training_error += squared_error(training_data[j]+out_ind);
            backward(training_data[j]+out_ind);
        }
        for(j=0; j<valid_r; j++){
            forward(validation_data[j]);
            validation_error += squared_error(validation_data[j]+out_ind);
        }
        training_error = training_error/train_r;
        validation_error = validation_error/valid_r;
        std::cout << training_error << "\t\t" << validation_error << std::endl;
        fp << i << "," << training_error << "," << validation_error << std::endl;
        export_weights(&i, &mode);
        adj_weights(ka, p, t, m);

        if(i==0){
            min = validation_error;
            mode = 1;
            export_weights(&i, &mode);
        }else{
            if(validation_error < min){
                min = validation_error;
                mode = 1;
                export_weights(&i, &mode);
            }
        }
        mode = 0;
    }

    fp.close();
};

// Method for measuring the squared error after a forward propagation
// The t parameter refers to the an array of the expected values that the output layer should have
double Neural_net::squared_error(double* t){
    int i;
    double error = 0;
    for(i=0; i<numONodes; i++){
        error += pow(z[i] - t[i], 2);
    }
    return error;
};

// Method for adjusting the weights
// The ka parameter refers to the kappa value
// The p parameter refers to the phi value
// The t parameter refers to the theta value
// The m parameter refers to the mu value
void Neural_net::adj_weights(double* ka, double* p, double* t, double* mu){
    int i, j, k;
    for(i=0; i<numHLayer; i++){
        for(j=0; j<numHNodes; j++){
            if(i==0){
                for(k=0; k<(numINodes+1); k++){
                    dhid[i][j][k] = dhid[i][j][k]/train_r;
                    if(fhid[i][j][k] * dhid[i][j][k] > 0){
                        ehid[i][j][k] += *ka;
                    }
                    if(fhid[i][j][k] * dhid[i][j][k] < 0){
                        ehid[i][j][k] = (*p) * ehid[i][j][k];
                    }
                    
                    fhid[i][j][k] = (1 - (*t)) * dhid[i][j][k] + (*t) * fhid[i][j][k];
                    c = *mu * c_hid[i][j][k] - (1 - *mu) * ehid[i][j][k] * dhid[i][j][k];
                    c_hid[i][j][k] = c;
                    hid[i][j][k] += c;
                    dhid[i][j][k] = 0;
                }
            }else{
                for(k=0; k<(numHNodes+1); k++){
                    dhid[i][j][k] = dhid[i][j][k]/train_r;
                    if(fhid[i][j][k] * dhid[i][j][k] > 0){
                        ehid[i][j][k] += *ka;
                    }
                    if(fhid[i][j][k] * dhid[i][j][k] < 0){
                        ehid[i][j][k] = (*p) * ehid[i][j][k];
                    }
                    
                    fhid[i][j][k] = (1 - (*t)) * dhid[i][j][k] + (*t) * fhid[i][j][k];
                    c = *mu * c_hid[i][j][k] - (1 - *mu) * ehid[i][j][k] * dhid[i][j][k];
                    c_hid[i][j][k] = c;
                    hid[i][j][k] += c;
                    dhid[i][j][k] = 0;
                }
            }
        }
    }

    for(i=0; i<numONodes; i++){
        for(j=0;j<(numHNodes+1); j++){
            dout[i][j] = dout[i][j]/train_r;
            if(fout[i][j] * dout[i][j] > 0){
                eout[i][j] += *ka;
            }
            if(fout[i][j] * dout[i][j] < 0){
                eout[i][j] = (*p) * eout[i][j];
            }

            fout[i][j] = (1 - (*t)) * dout[i][j] + (*t) * fout[i][j];
            c = *mu * c_out[i][j] - (1 - *mu) * (eout[i][j]) * dout[i][j];
            c_out[i][j] = c;
            out[i][j] += c;
            dout[i][j] = 0;
        }
    }
}

// Method for classifying a particular array of input values 
// The inp parameter refers to the array of input values
int Neural_net::classify(double* inp){
    int i, genre, multiplier;    
    forward(inp);
    genre = 0;
    multiplier = 1;
    for(i=(numONodes-1); i>-1; i--){
        if(z[i] >= 0.50) genre += multiplier;
        multiplier *= 2;
    }
    return genre;
}

// Method for classifying the genre of the input values in the testing dataset
// It also exports the classified genre and the expected genre to the test_results11.csv file
// Also prints the accuracy of the neural network
// Accuracy = (Number of equal matches between the classified genre and the expected genre)/(total number rows in the testing dataset)
void Neural_net::generate_class(){
    std::ofstream fp;
    int i;
    double count = 0;
    fp.open("test_results11.csv");
    for(i=0; i<test_r; i++){
        fp << classify(testing_data[i]) << "," << testing_data[i][51] << std::endl;
        if(classify(testing_data[i]) == testing_data[i][51]) count++;
    }
    std::cout << "Accuracy: " << count/test_r << std::endl;
    fp.close();
}