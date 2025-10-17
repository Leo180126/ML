#include "../include/utils.h"
double randn (double mu, double sigma){
    static double X2;
    static int call = 0;
    double U1, U2, W, mult;

    if (call) {
        call = 0;
        return mu + sigma * X2;
    }

    do {
        U1 = -1.0 + 2.0 * ((double) rand() / RAND_MAX);
        U2 = -1.0 + 2.0 * ((double) rand() / RAND_MAX);
        W = U1 * U1 + U2 * U2;
    } while (W >= 1.0 || W == 0.0);

    mult = sqrt((-2.0 * log(W)) / W);
    X2 = U2 * mult;
    call = 1;
    return mu + sigma * (U1 * mult);
}
