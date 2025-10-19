#include "../include/linear_regression.h"
void initModel(LinearRegression *p, int numOfFeatures, const char* typeOfInit){
    init2DTensor(&(p->weights), numOfFeatures + 1, 1);
    p->numOfFeatures = numOfFeatures;
    p->weights.data[0] = 0;
    if(strcmp(typeOfInit, "zero") == 0){
        for(int i=1; i<numOfFeatures + 1; i++){
            p->weights.data[i] = 0;
        }
    }
    else if(strcmp(typeOfInit, "normal") == 0){
        for(int i=1; i<numOfFeatures + 1; i++){
            p->weights.data[i] = randn(0, 1);
        }
    }
    else if(strcmp(typeOfInit, "uniform") == 0){
        for(int i=1; i<numOfFeatures + 1; i++){
            p->weights.data[i] = -1 + ((double) rand () / RAND_MAX) * 2;
        }
    }
    p->coef_ = (double *)malloc(p->numOfFeatures*sizeof(double));
}
void fit(LinearRegression *p, Tensor *X, Tensor *y){
    insColTo2DTensor(X, 1, 0);
    Tensor w_T, y_hat, X_i, grad;
    init2DTensor(&(w_T), 1, 6);
    transMatrix(&(p->weights), &(w_T));
    init2DTensor(&grad, 1, 6);
    init2DTensor(&(X_i), 6, 1);
    init2DTensor(&(y_hat), 1, 1);
    int epochs = 10000;
    double lr = 0.1;
    double prev_cost;
    double cost;
    for(int epoch = 1; epoch <= epochs; epoch++){
        cost = 0;
        for(int i=0, label_index=0; i<X->size; i += 6, label_index++){
            assignTensorFromMemory(&X_i, address(X, i));
            multiMatrix(&y_hat, &w_T, &X_i);
            cost += (y_hat.data[0] - y->data[label_index])*(y_hat.data[0] - y->data[label_index]);
            mutilScalarToTensor(&X_i, y_hat.data[0] - y->data[label_index]);
            addTensor(&grad, &grad, &X_i);
        }
        mutilScalarToTensor(&grad, -1.0*lr*(1.0/X->size));
        printf("\n%lf", cost);
        addTensor(&w_T, &w_T, &grad);
        if(epoch > 1 && fabs(cost - prev_cost) < 1e-12){
            break;
        }
        prev_cost = cost;
    }
    assignTensorFromMemory(&(p->weights), w_T.data);
    p->intercept_ = p->weights.data[0];
    for(int i=0; i<p->numOfFeatures; i++){
        p->coef_[i] = p->weights.data[i+1];
    }
    freeTensor(&(w_T));
    freeTensor(&(y_hat));
    freeTensor(&(X_i));
    freeTensor(&grad);
    
}
void freeModel(LinearRegression *p){
    freeTensor(&(p->weights));
    free(p->coef_);
}
void printBias(LinearRegression *p){
    printf("\nBias: ");
    printf("%lf", p->intercept_);
}
void printWeight(LinearRegression *p){
    printf("\nWeight: ");
    for(int i=0; i<p->numOfFeatures; i++){
        printf("%lf ", p->coef_[i]);
    }
}
void MSEScore(LinearRegression *p, Tensor X_test, Tensor y_test){
    insColTo2DTensor(&X_test, 1, 0);
    double lost = 0;
    Tensor y_hat, X_i;
    init2DTensor(&y_hat, 1, 1);
    init2DTensor(&X_i, 1, 6);
    int label_index = 0;
    for(int i=0; i<X_test.size; i+=6){
        label_index++;
        assignTensorFromMemory(&X_i, address(&X_test, i));
        multiMatrix(&y_hat, &X_i, &p->weights);
        lost += (y_hat.data[0] - y_test.data[label_index])*(y_hat.data[0] - y_test.data[label_index]);
    }
    lost /= X_test.size;
    printf("\nMSE of test set: %lf", lost);
}