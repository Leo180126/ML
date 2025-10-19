#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H
#include <stdio.h>
#include "tensor.h"
#include "utils.h"
typedef struct
{
    Tensor weights;
    int numOfFeatures;
}LogisticRegression;
void initModel(LogisticRegression *model, int numOfFeatures, const char *typeOfInit);
void fit(LogisticRegression *model, Tensor X, Tensor y);
void freeModel(LogisticRegression *model);
#endif