#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include <string.h>
#include "utils.h"
#include "tensor.h"
typedef struct 
{
    Tensor weights;
    int numOfFeatures;
    double *coef_;
    double intercept_;
}LinearRegression;
void initModel(LinearRegression *p, int numOfFeatures, const char* typeOfInit);
void fit(LinearRegression *p, Tensor *X, Tensor *y);
void freeModel(LinearRegression*p);
void printBias(LinearRegression *p);
void printWeight(LinearRegression *p);
void MSEScore(LinearRegression *p, Tensor X_test, Tensor y_test);
#endif