#include "../include/logistic_regression.h"
void initModel(LogisticRegression *model, int numOfFeatures, const char *typeOfInit){
    model->numOfFeatures = numOfFeatures;
    init2DTensor(&(model->weights), numOfFeatures + 1, 1);
    model->weights.data[0] = 0;
    if(strcmp(typeOfInit, "zero") == 0){
        for(int i=1; i<numOfFeatures + 1; i++){
            model->weights.data[i] = 0;
        }
    }
    else if(strcmp(typeOfInit, "normal") == 0){
        for(int i=1; i<numOfFeatures + 1; i++){
            model->weights.data[i] = randn(0, 1);
        }
    }
    else if(strcmp(typeOfInit, "uniform") == 0){
        for(int i=1; i<numOfFeatures + 1; i++){
            model->weights.data[i] = -1 + ((double) rand () / RAND_MAX) * 2;
        }
    }
}
void fit(LogisticRegression *model, Tensor X, Tensor y){
    insColTo2DTensor(&X, 1, 0);
    Tensor s, X_i, gradTensor;
    init2DTensor(&s, 1, 1);
    init2DTensor(&X_i, 1, X.dim[0]);
    init2DTensor(&gradTensor, X.dim[0], 1);
    assignTensorWithNum(&gradTensor, 0.0);
    double L = 0;
    int label_index = 0;
    int epochs = 100;
    double lr = 0.0001;
    double grad;
    for(int epoch = 1; epoch <= epochs; epoch++){
        grad = 0;
        for(int i=0; i<X.size; i+=9){
            assignTensorFromMemory(&X_i, address(&X, i));
            multiMatrix(&s, &X_i, &model->weights);
            double sigmoid = 1/(1+exp(s.data[0]));
            L += y.data[label_index]*log(sigmoid) + (1-y.data[label_index])*log(1-sigmoid);
            grad += (y.data[label_index] - sigmoid);
        }
        addScalarToTensor(&gradTensor, grad/X.size); 
        elementWiseMutil(&gradTensor, &gradTensor, &X_i);
        mutilScalarToTensor(&gradTensor, lr);
        addTensor(&model->weights, &model->weights, &gradTensor);
        printf("\nLoss: %lf", L);
        label_index++;
    }
    freeTensor(&s);
    freeTensor(&X_i);
    freeTensor(&gradTensor);
}
void freeModel(LogisticRegression *model){
    freeTensor(&model->weights);
}
