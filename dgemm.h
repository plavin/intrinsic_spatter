#ifndef DGEMM_H
#define DGEMM_H

#define DGEMM_DIM 1024l
#define DGEMM_BLOCK 32

double dgemm();
double dgemm_mflops(double time_s);

#endif
