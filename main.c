#include <immintrin.h>
#include <stdio.h>
#include <stdio.h>
#include <assert.h>

int main() {

    int scale = 1;
    int align = 4096;
    __m256i vindex = _mm256_set_epi32(12, 1, 2, 3, 4, 5, 6, 7);

    double *src = (double *)aligned_alloc(align, sizeof(double) * 128);
    for (int i = 0; i < 128; i++) {
        src[i] = i;
    }

    __m512d dst = _mm512_i32gather_pd(vindex, src, 1);

  double* f = (double*)&dst;
  /*
  printf("%.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf %.1lf\n",
    f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    */
  printf("\n\n%lf\n", f[0]);
  //printf("%lf\n", f[1]);

  return 0;
}
