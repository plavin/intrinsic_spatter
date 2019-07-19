#include <immintrin.h>
#include <stdio.h>
#include <stdio.h>
#include <assert.h>

#define ALIGN 4096

int main() {

    int src_len = 128;
    __m256i vindex = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int *bb = (int*)&vindex;
    for (int i = 0; i < 8; i++) {
        printf("%d ", bb[i]);
    }
    printf("\n");

    double *src = (double *)aligned_alloc(ALIGN, sizeof(double) * src_len);
    for (int i = 0; i < src_len; i++) {
        src[i] = i;
    }

    //__m512d dst = _mm512_set4_pd(8,9,8,9);
    //__m512d dst = _mm512_setzero_pd();
    __m512d dst = _mm512_i32gather_pd(vindex, src, 1);

  double* f = (double*)&dst;
  //printf("%.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf\n",f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
  printf("%lf\n", f[1]);
  printf("%lf\n", f[0]);

  return 0;
}
