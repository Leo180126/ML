#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
typedef struct{
    double *data;
    int *dim;                                             // Luu tu cot -> hang
    int size;
    int numOfDim;
}Tensor;
void initTensor(Tensor *p, const char *dim);              // Dim tu thap den cao, so cot -> so hang
void init2DTensor(Tensor *p, int row, int col);           // 2D only
double tensor2D(Tensor *p, int dim1, int dim0);
double tensor3D(Tensor *p, int dim2, int dim1, int dim0);
double tensor1D(Tensor *p, int dim0);
void insTo1DTensor(Tensor *p, double var, int des);       // Them vao tai des va day lui phan sau
void insColTo2DTensor(Tensor *p, double var, int des);
void printTensor(Tensor *p);
void freeTensor(Tensor *p);
int tensorCmp(Tensor *a, Tensor *b);
void addTensor(Tensor *ans, Tensor *a, Tensor *b);
void assignTensor(Tensor *p, const char *para);
void multiMatrix(Tensor *ans, Tensor *a, Tensor *b);
Tensor *transMatrix(Tensor *p, Tensor *p_t);
void assignTensorFromMemory(Tensor *p, double *a);
double *address(Tensor *p, int index);
void dimOfTensor(Tensor *p);
void addScalarToTensor(Tensor *p, double var);
void mutilScalarToTensor(Tensor *p, double var);
#endif