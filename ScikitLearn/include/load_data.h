#ifndef LOAD_DATA_H
#define LOAD_DATA_H
#include <stdio.h>
#include <string.h>
#include "utils.h"
#include "tensor.h"
#define MAX_LINE_LENGTH 2048
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
}DataSet;
void loadData(DataSet *p, const char *filename);
void initDataSet(DataSet *p, int numOfExample, int numOfFeature);
void printData(DataSet *p);
void freeDataSet(DataSet *p);
void createTensor(DataSet *p);
void standerlizeData(DataSet *ds);
void splitTrainTest(DataSet *p, double testPercent);
#endif