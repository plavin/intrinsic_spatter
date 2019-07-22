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



void gather(int src_len, double *src, __m512d *dst, __m256i vindex);

int main() {

    int src_len = 1<<29; //2**29 doubles == 4GB
    double *src;
    __m256i vindex; 
    __m512d dst; 
    // Set vindex to read the first 8 elements of the bufer
    vindex = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);

#ifdef __INTEL_COMPILER
    src = (double *)_mm_malloc(sizeof(double) * src_len, ALIGN);
#else //Assume GCC
    src = (double *)aligned_alloc(ALIGN, sizeof(double) * src_len);
#endif
    assert(src);
    for (int i = 0; i < src_len; i++) {
        src[i] = i;
    }

    sg_zero_time();
    gather(src_len, src, &dst, vindex);
    double time_ms = sg_get_time_ms();
    double time_s = time_ms / 1000;

    printf("BW(MB/s): %lf\n", src_len*sizeof(double)/1000/1000 / time_s);

  double* f = (double*)&dst;
  printf("%.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf\n",f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

#ifdef __INTEL_COMPILER
    _mm_free(src);
#else //Assume GCC
    free(src);
#endif
  return 0;
}

void gather(int src_len, double *src, __m512d *dst, __m256i vindex)
{
#pragma ivdep
    for (int i = 0; i < src_len / 8; i++) {
        *dst = _mm512_i32gather_pd(vindex, src+i*8, 8);
    }
}
