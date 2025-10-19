#include "../include/load_data.h"
void loadData(DataSet *p, const char *filename){
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Khong mo duoc file");
    }
    char line[MAX_LINE_LENGTH];
    int row = 0;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp) && row < p->numOfExample){
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < p->numOfCols){
            if(strcmp(token, "Yes") == 0)  p->data[row*p->numOfCols + col] = 1;
            else if(strcmp(token, "No") == 0) p->data[row*p->numOfCols + col] = 0;
            else p->data[row*p->numOfCols + col] = atoi(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    fclose(fp);
}
void initDataSet(DataSet *p, int numOfExample, int numOfFeature){
    p->data = (double *)malloc(numOfExample*(numOfFeature+1)*sizeof(double));
    p->numOfExample = numOfExample;
    p->numOfFeature = numOfFeature;
    p->numOfCols = numOfFeature + 1;
    p->trainSize = p->numOfExample;
    p->testSize = 0;
}
void printData(DataSet *p){
    printf("\n");
    for(int row = 0; row<p->numOfExample; row++){
        for(int col = 0; col<p->numOfCols; col++){
            printf("%lf ", p->data[row*(p->numOfFeature+1) + col]);
        }
        printf("\n");
    }
    printf("\n");
}
void freeDataSet(DataSet *p){
    free(p->data);
    freeTensor(&(p->train));
    freeTensor(&(p->label));
    freeTensor(&(p->test));
    freeTensor(&(p->test_label));
    free(p->mean);
    free(p->std);
}
void splitTrainTest(DataSet *p, double testPercent){
    p->trainSize = p->numOfExample*(1-testPercent);
    p->testSize = p->numOfExample - p->trainSize;
}

void createTensor(DataSet *p){
    init2DTensor(&(p->label), p->trainSize, 1);
    init2DTensor(&(p->train), p->trainSize, p->numOfCols-1);
    init2DTensor(&(p->test_label), p->testSize, 1);
    init2DTensor(&(p->test), p->testSize, p->numOfCols-1);
    int train_index = 0;
    int test_index = 0;
    int label_index = 0;
    int test_label_index = 0;
    for(int i=0; i<p->numOfExample; i++){
        for(int j=0; j<p->numOfCols; j++){
            if(i < p->trainSize){
                if(j < p->numOfCols - 1){
                    p->train.data[train_index++] = p->data[i * p->numOfCols + j];
                }
                else{
                    p->label.data[label_index++] = p->data[i * p->numOfCols + j];
                }
            }
            else{
                if (j < p->numOfCols - 1)
                    p->test.data[test_index++] = p->data[i * p->numOfCols + j];
            else
                    p->test_label.data[test_label_index++] = p->data[i * p->numOfCols + j];
            }
        }
    }
}
void standerlizeData(DataSet *ds){
    int m = ds->numOfFeature;
    int n = ds->numOfExample;
    ds->mean = (double *)calloc(m, sizeof(double));
    ds->std = (double *)calloc(m, sizeof(double));
    for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
            ds->mean[j] += ds->data[i*ds->numOfCols + j];
        }
        ds->mean[j] /= n;   
    }
    for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
            double diff = ds->data[i*ds->numOfCols + j] - ds->mean[j];
            ds->std[j] += diff*diff;
        }
        ds->std[j] = sqrt(ds->std[j]/n);
        if(ds->std[j] < 1e-12) ds->std[j] = 1.0;
    }
    for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
            ds->data[i*ds->numOfCols + j] = (ds->data[i*ds->numOfCols + j] - ds->mean[j])/ds->std[j];
        }
    }
}