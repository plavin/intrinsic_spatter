#include <stdlib.h>
#include <assert.h>
#include "dgemm.h"

double dgemm() {

    double *A = (double *)malloc(sizeof(double) * DGEMM_DIM * DGEMM_DIM);
    double *B = (double *)malloc(sizeof(double) * DGEMM_DIM * DGEMM_DIM);
    double *C = (double *)malloc(sizeof(double) * DGEMM_DIM * DGEMM_DIM);
    assert(A); assert(B); assert(C);

    int rand1 = rand() % DGEMM_DIM;
    int rand2 = rand() % DGEMM_DIM;

    for (long i = 0; i < DGEMM_DIM * DGEMM_DIM; i++) {
       A[i] = rand1;
       B[i] = rand2;
    }


    int en = DGEMM_BLOCK * (DGEMM_DIM / DGEMM_BLOCK);
    for (int kk = 0; kk < en; kk += DGEMM_BLOCK) {
        for (int jj = 0; jj < en; jj += DGEMM_BLOCK) {
            for (int i = 0; i < DGEMM_DIM; i++) {
               for (int j = jj; j < jj+DGEMM_BLOCK; j++) {
                   double sum = C[i*DGEMM_DIM+j];
                   for (int k = kk; k < kk+DGEMM_BLOCK; k++) {
                        sum += A[i*DGEMM_DIM+k] * B[k*DGEMM_DIM+j];
                   }
                   C[i*DGEMM_DIM+j] = sum;
               }
            }
        }
    }

    double ret = C[rand1*DGEMM_DIM + rand2];

    free(A);
    free(B);
    free(C);

    return ret;

}

double dgemm_mflops(double time_s) {
    return ((double)DGEMM_DIM * DGEMM_DIM * DGEMM_DIM)*2/1000/1000 / time_s;
}
