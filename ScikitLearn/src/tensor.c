#include "../include/tensor.h"
void initTensor(Tensor *p, const char *dim){
    
    int i=0, count = 0;
    while(dim[i] != '\0'){
        if(dim[i] == ','){
            count++;
        }
        i++;
    }
    p->dim = (int *)malloc((count + 1)*sizeof(int));
    p->numOfDim = count+1;

    int num = 0;
    i=0;
    while(*dim){
        if (*dim >= '0' && *dim <= '9') {
            num = num * 10 + (*dim - '0');
        } else if (num > 0) {
            p->dim[i] = num;
            i++;
            num = 0;
        }
        dim++;
    }
    if(num > 0) p->dim[i] = num;

    int a=0, b=count;
    while(a<b){
        int temp = p->dim[a];
        p->dim[a] = p->dim[b];
        p->dim[b] = temp;
        a++;
        b--;
    }

    p->size = 1;
    for(int i=0; i<count+1; i++){
        p->size *= p->dim[i];
    }

    p->data = (double *)malloc(p->size*sizeof(double));
}
double tensor2D(Tensor *p, int dim1, int dim0){
    if(p->numOfDim != 2){
        printf("Not a matrix");
        return 0;
    }
    return p->data[dim1*p->dim[0]+dim0];
}
double tensor3D(Tensor *p, int dim2, int dim1, int dim0){
    if(p->numOfDim != 3){
        printf("Not a tensor");
        return 0;
    }
    return p->data[dim2*p->dim[1]+dim1*p->dim[0]+dim0];
}
double tensor1D(Tensor *p, int dim0){
    if(p->numOfDim != 1){
        printf("Not a vector");
        return 0;
    }
    return p->data[dim0];
}
void freeTensor(Tensor *p){
    free(p->dim);
    free(p->data);
}
void printTensor(Tensor *p){
    if(p->numOfDim == 1){
        printf("\n Vector %p :\n", p);
        for(int i=0; i<p->size; i++){
            printf("%lf ", tensor1D(p, i));
        }
    }
    if(p->numOfDim == 2){
        printf("\n Matrix %p :\n", p);
        for(int i=0; i<p->dim[1]; i++){
            for(int j=0; j<p->dim[0]; j++){
                printf("%lf ", tensor2D(p, i, j));
            }
            printf("\n");
        }
    }
    if(p->numOfDim == 3){
        printf("\n Tensor %p :\n", p);
        for(int i=0; i<p->dim[2]; i++){
            for(int j=0; j<p->dim[1]; j++){
                for(int k=0; k<p->dim[0]; k++){
                    printf("%lf ", tensor3D(p, i, j, k));
                }
                printf("\n");
            }
            printf("\n");
            printf("\n");
        }
    }
}
void init2DTensor(Tensor *p, int row, int col){
    p->numOfDim = 2;
    p->dim = (int *)malloc(2*sizeof(int));
    p->dim[1] = row;
    p->dim[0] = col;
    p->size = row*col;
    p->data = (double *)malloc(p->size*sizeof(double));
}
void insTo1DTensor(Tensor *p, double var, int des){
    p->size++;
    p->dim[0]++;
    double* temp = (double *)realloc(p->data, p->size);
    if(temp == NULL){
        perror("realloc failed");
        exit(EXIT_FAILURE);
    }
    p->data = temp;
    
    for(int i=p->size-1; i>des; i--){
        p->data[i] = p->data[i-1];
    }
    p->data[des] = var; 
}
void insColTo2DTensor(Tensor *p, double var, int des){
    p->size += p->dim[1];
    p->dim[0]++;

    double *temp = realloc(p->data, p->size*sizeof(double));
    if(temp == NULL){
        perror("realloc failed");
        exit(EXIT_FAILURE);
    }
    p->data = temp;

    temp = (double *)malloc(p->size*sizeof(double));
    for(int i=0; i<p->size; i++){
        temp[i] = p->data[i];
    }
    int dem = 0;
    for(int i=0; i<p->dim[1]; i++){
        for(int j=0; j<p->dim[0]; j++){
            if(j == des){
                p->data[i*p->dim[0] + j] = var;
            }
            else{
                p->data[i*p->dim[0] + j] = temp[dem++];
            }
        }
    }
    free(temp);
}
int tensorCmp(Tensor *a, Tensor *b){
    int cmp = a->numOfDim - b->numOfDim;
    if(cmp == 0){
        for(int i=0; i<a->numOfDim; i++){
            if(a->dim[i] != b->dim[i]){
                cmp = a->dim[i] - b->dim[i];
                return cmp;
            }
        }
    }
    return cmp;
}
void addTensor(Tensor *ans, Tensor *a, Tensor *b){
    // if(tensorCmp(a, b) != 0){
        // printf("Hai tensor khac kich thuoc");
        // return;
    // }
    if(a->size != b->size){
        printf("Hai tensor khac kich thuoc");
        return;
    }
    for(int i=0; i<a->size; i++){
        ans->data[i] = a->data[i] + b->data[i];
    }
}

void assignTensor(Tensor *p, const char *para){
    if(strcmp(para, "zero") == 0){
        for(int i=0; i<p->size; i++){
            p->data[i] = 0;
        }
    }
    else if(strcmp(para, "normal") == 0){
        for(int i=0; i<p->size; i++){
            p->data[i] = randn(0, 0.1);
        }
    }
    else if(strcmp(para, "uniform") == 0){
        for(int i=0; i<p->size; i++){
            p->data[i] = -1 + ((double) rand () / RAND_MAX) * 2;
        }
    }
}
void multiMatrix(Tensor *ans, Tensor *a, Tensor *b){
    if(a->size != b ->size){
        printf("Nhan 2 tensor khong cung kich thuoc");
        return;
    }
    if(a->dim[0] != b->dim[1]){
        printf("Khong the nhan [%d, %d] cho [%d, %d]", a->dim[0], a->dim[1], b->dim[0], b->dim[1]);
        return;
    }
    for(int i=0; i<a->dim[1]; i++){
        for(int j=0; j<b->dim[0]; j++){
            double sum = 0.0;
            for(int k=0; k<a->dim[0]; k++){
                sum += a->data[i*a->dim[0] + k] * b->data[k*b->dim[0] + j];
            }
            ans->data[i*ans->dim[0] + j] = sum;
        }
    }
}
Tensor* transMatrix(Tensor *p, Tensor *p_t){
    init2DTensor(p_t, p->dim[0], p->dim[1]);
    for(int i=0; i<p->dim[1]; i++){
        for(int j=0; j<p->dim[0]; j++){
            p_t->data[j*p_t->dim[0] + i] = p->data[i*p->dim[0]+j];
        }
    }
    return p_t;
}
void assignTensorFromMemory(Tensor *p, double *a){
    for(int i=0; i<p->size; i++){
        p->data[i] = a[i];
    }
}
double *address(Tensor *p, int index){
    return p->data + index;
}
void dimOfTensor(Tensor *p){
    printf("\nNum of dim (%d) :", p->numOfDim);
    for(int i=p->numOfDim - 1; i>=0; i--){
        printf("%d ", p->dim[i]);
    }
    printf("\n");
}
void addScalarToTensor(Tensor *p, double var){
    for(int i=0; i<p->size; i++){
        p->data[i] += var;
    }
}
void mutilScalarToTensor(Tensor *p, double var){
    for(int i=0; i<p->size; i++){
        p->data[i] *= var;
    }
}