#ifndef DIABETES_H
#define DIABETES_H
#include "tensor.h"
typedef struct{
    double *data;
    Tensor label;
    Tensor test_label;
    Tensor train;
    Tensor test;
    int numOfCols;
    int testSize;
    int trainSize;
    int numOfFeature;
    int numOfExample;
    double *mean;
    double *std;
}DiabetesData;

#endif