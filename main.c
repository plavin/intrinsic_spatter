#include <immintrin.h>
#include <stdio.h>
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include "sgtime.h"
#ifdef __INTEL_COMPILER
#include <malloc.h>
#endif

#define ALIGN 4096
#define src_len (1<<14)
#define DIM 1024l
#define BLOCK 32

void gather(int, double *src, __m512d *dst, __m512d *dst2, __m512d *dst3, __m512d *dst4, __m256i vindex);
void gather8(int _src_len, double *src, __m512d *dst, __m256i vindex);
void gatherms1(int _src_len, double *src, __m512d *dst,__m512d *dst2,  __m512d *dst3, __m512d *dst4,__m256i vindex);

void set256(__m256i *index) {
    *index = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
}
void set256ms1(__m256i *index) {
    *index = _mm256_set_epi32(0, 1, 2, 3, 8, 9, 10, 11);
}
void set256ms12(__m256i *index) {
    *index = _mm256_set_epi32(0, 1, 8, 9, 16, 17, 24, 25);
}
void set256ms13(__m256i *index) {
    *index = _mm256_set_epi32(0, 8, 16, 24, 32, 40, 48, 56);
}

double dgemm() {

    double *A = (double *)malloc(sizeof(double) * DIM * DIM); 
    double *B = (double *)malloc(sizeof(double) * DIM * DIM); 
    double *C = (double *)malloc(sizeof(double) * DIM * DIM); 
    assert(A); assert(B); assert(C);

    int rand1 = rand() % DIM;
    int rand2 = rand() % DIM;

    for (long i = 0; i < DIM * DIM; i++) {
       A[i] = rand1;
       B[i] = rand2;
    }

    
    int en = BLOCK * (DIM / BLOCK);
    for (int kk = 0; kk < en; kk += BLOCK) {
        for (int jj = 0; jj < en; jj += BLOCK) {
            for (int i = 0; i < DIM; i++) {
               for (int j = jj; j < jj+BLOCK; j++) {
                   double sum = C[i*DIM+j];
                   for (int k = kk; k < kk+BLOCK; k++) {
                        sum += A[i*DIM+k] * B[k*DIM+j];
                   }
                   C[i*DIM+j] = sum;
               }
            }
        }
    }

    double ret = C[rand1*DIM + rand2];

    free(A);
    free(B);
    free(C);
    
    return ret;

}

int main() {

    int ms1_mode = 1;
    int ntimes = 1000;
    int use8 = 0;
    //int src_len = 1<<29; //2**29 doubles == 4GB
    double *src;
    __m256i vindex; 
    __m512d dst, dst2, dst3, dst4;
    __m512d dst8[8];
    // Set vindex to read the first 8 elements of the bufer
    //vindex = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    if (ms1_mode) {
        set256ms1(&vindex);
    } else {
        set256(&vindex);
    }
    printf("source length: %d\n", src_len);

#ifdef __INTEL_COMPILER
    src = (double *)_mm_malloc(sizeof(double) * src_len, ALIGN);
#else //Assume GCC
    src = (double *)aligned_alloc(ALIGN, sizeof(double) * src_len);
#endif
    assert(src);

    for (int i = 0; i < src_len; i++) {
        src[i] = i;
    }

    for (int i = 0; i < 3; i++) {
        sg_zero_time();
        printf("Val: %lf\n", dgemm());
        double time_dgemm_s = sg_get_time_ms() / 1000;
        long mflops = DIM*DIM*DIM*2 / 1000 / 1000;
        printf("DGEMM: %lf seconds, %lf MFlops\n", time_dgemm_s, mflops/time_dgemm_s);
    }
    //cache warm
    if (ms1_mode) {
        gatherms1(src_len, src, &dst, &dst2, &dst3, &dst4, vindex);
    } else {
        gather(src_len, src, &dst, &dst2, &dst3, &dst4, vindex);
    }

    sg_zero_time();
    for (int i = 0; i < ntimes; i++) {
        if (ms1_mode) {
            gatherms1(src_len, src, &dst, &dst2, &dst3, &dst4, vindex);
        } else {
            if (use8) {
                gather8(src_len, src, dst8, vindex);
            } else {
                gather(src_len, src, &dst, &dst2, &dst3, &dst4, vindex);
            }
        }
    }
    double time_ms = sg_get_time_ms()/ntimes;
    double time_s = time_ms / 1000;
    
    if (ms1_mode) {
        printf("BW(MB/s): %lf\n", ((double)src_len)*sizeof(double)/1000/1000/2 / time_s);
    } else {
        printf("BW(MB/s): %lf\n", ((double)src_len)*sizeof(double)/1000/1000 / time_s);
    }


  double* f = (double*)&dst;
  printf("%.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf\n",f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

#ifdef __INTEL_COMPILER
    _mm_free(src);
#else //Assume GCC
    free(src);
#endif
  return 0;
}

void gather(int _src_len, double *src, __m512d *dst,__m512d *dst2,  __m512d *dst3, __m512d *dst4,__m256i vindex)
{
//#pragma ivdep
    for (int i = 0; i < src_len / 8; i+=4) {

        //GOOD
        //*dst  = _mm512_i32gather_pd(vindex, src, 8);
        //src += 8;

        //BAD
        // NOTE - You must use i*8*8 for default optimization with gcc, 
        // NOTE - and you should use i*8 at -O3 ...
        //*dst  = _mm512_i32gather_pd(vindex, src+i*8*8, 8);
        *dst  = _mm512_i32gather_pd(vindex, src+i*8, 8);
        *dst2 = _mm512_i32gather_pd(vindex, src+(i+1)*8, 8);
        *dst3 = _mm512_i32gather_pd(vindex, src+(i+2)*8, 8);
        *dst4 = _mm512_i32gather_pd(vindex, src+(i+3)*8, 8);

        //IGNORE
        //*dst  = _mm512_i32gather_pd(vindex, src+i*8, 8);
        //*dst2 = _mm512_i32gather_pd(vindex, src+(i+1)*8, 8);
        //*dst3 = _mm512_i32gather_pd(vindex, src+(i+2)*8, 8);
        //*dst4 = _mm512_i32gather_pd(vindex, src+(i+3)*8, 8);
    }

}

void gather8(int _src_len, double *src, __m512d *dst, __m256i vindex)
{
//#pragma ivdep
    for (int i = 0; i < src_len / 8; i+=8) {

        //GOOD
        //*dst  = _mm512_i32gather_pd(vindex, src, 8);
        //src += 8;

        //BAD
        // NOTE - You must use i*8*8 for default optimization with gcc, 
        // NOTE - and you should use i*8 at -O3 ...
        //*dst  = _mm512_i32gather_pd(vindex, src+i*8*8, 8);
        dst[0] = _mm512_i32gather_pd(vindex, src+i*8, 8);
        dst[1] = _mm512_i32gather_pd(vindex, src+(i+1)*8, 8);
        dst[2] = _mm512_i32gather_pd(vindex, src+(i+2)*8, 8);
        dst[3] = _mm512_i32gather_pd(vindex, src+(i+3)*8, 8);
        dst[4] = _mm512_i32gather_pd(vindex, src+(i+4)*8, 8);
        dst[5] = _mm512_i32gather_pd(vindex, src+(i+5)*8, 8);
        dst[6] = _mm512_i32gather_pd(vindex, src+(i+6)*8, 8);
        dst[7] = _mm512_i32gather_pd(vindex, src+(i+7)*8, 8);

        //IGNORE
        //*dst  = _mm512_i32gather_pd(vindex, src+i*8, 8);
        //*dst2 = _mm512_i32gather_pd(vindex, src+(i+1)*8, 8);
        //*dst3 = _mm512_i32gather_pd(vindex, src+(i+2)*8, 8);
        //*dst4 = _mm512_i32gather_pd(vindex, src+(i+3)*8, 8);
    }

}


void gatherms1(int _src_len, double *src, __m512d *dst,__m512d *dst2,  __m512d *dst3, __m512d *dst4,__m256i vindex)
{
#pragma ivdep
    for (int i = 0; i < src_len / 16; i+=4) {
        *dst  = _mm512_i32gather_pd(vindex, src+i*16, 8);
        *dst2 = _mm512_i32gather_pd(vindex, src+(i+1)*16, 8);
        *dst3 = _mm512_i32gather_pd(vindex, src+(i+2)*16, 8);
        *dst4 = _mm512_i32gather_pd(vindex, src+(i+3)*16, 8);
    }
}
