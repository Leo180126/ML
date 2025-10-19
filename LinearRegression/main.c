#include "../ScikitLearn/include/load_data.h"
#include "../ScikitLearn/include/utils.h"
#include "../ScikitLearn/include/tensor.h"
#include "../ScikitLearn/include/linear_regression.h"
int main(){
    srand(time(NULL));
    
    DataSet ds;
    initDataSet(&ds, 10000, 5);
    loadData(&ds, "../data/Student_Performance.csv");
    standerlizeData(&ds);
    splitTrainTest(&ds, 0.2);
    createTensor(&ds);
    printData(&ds);
    LinearRegression model;
    initModel(&model, 5, "normal");
    fit(&model, &(ds.train), &(ds.label));
    printBias(&model);
    printWeight(&model);
    MSEScore(&model, ds.test, ds.test_label);
    freeDataSet(&ds);
    freeModel(&model);
}