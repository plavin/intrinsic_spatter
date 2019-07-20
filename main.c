#include <immintrin.h>
#include <stdio.h>
#include <stdio.h>
#include <assert.h>
#ifdef __INTEL_COMPILER
#include <malloc.h>
#endif

#define ALIGN 4096

int main() {

    int src_len = 128;
    double *src;
    __m256i vindex; 
    
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

    __m512d dst = _mm512_i32gather_pd(vindex, src, 8);

  double* f = (double*)&dst;
  printf("%.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf\n",f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

#ifdef __INTEL_COMPILER
    _mm_free(src);
#else //Assume GCC
    free(src);
#endif
  return 0;
}
