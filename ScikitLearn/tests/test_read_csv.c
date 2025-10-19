#include "../include/load_data.h"
#include "../include/utils.h"
#include "../include/tensor.h"
#include "../include/logistic_regression.h"
int main(){
    srand(time(NULL));
    
    DataSet ds;
    initDataSet(&ds, 768, 8);
    loadData(&ds, "../data/diabetes.csv");
    standerlizeData(&ds);
    splitTrainTest(&ds, 0.2);
    createTensor(&ds);
    // printData(&ds);
    
    LogisticRegression model;
    initModel(&model, 8, "normal");
    fit(&model, ds.train, ds.label);

    freeModel(&model);
    freeDataSet(&ds);

}