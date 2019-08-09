#include <immintrin.h>
#include <stdio.h>
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include "sgtime.h"
#include "dgemm.h"
#ifdef __INTEL_COMPILER
#include <malloc.h>
#endif

#define ALIGN 4096
#define src_len (1<<14)

void gather(int, double *src, __m512d *dst, __m512d *dst2, __m512d *dst3, __m512d *dst4, __m256i vindex);
void gather_contig(int, double *src, __m512d *dst, __m512d *dst2, __m512d *dst3, __m512d *dst4);
void gather_contig8(int _src_len, double *src, __m512d *dst,__m512d *dst2,  __m512d *dst3, __m512d *dst4, __m512d *dst5, __m512d *dst6, __m512d *dst7, __m512d *dst8);
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


int main(int argc, char **argv) {

    int ms1_mode = 0;
    int contigload = 1;
    int ntimes = 100083;
    int use8 = 0;
    //int src_len = 1<<29; //2**29 doubles == 4GB
    double *src;
    __m256i vindex; 
    __m512d dst, dst2, dst3, dst4, dst5, dst6, dst7, dst8;
#ifndef __INTEL_COMPILER
    register __m512i z31 asm("zmm31") = _mm512_set1_epi32(123); 
#endif
    //__m512d dst8[8];
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

    int print_bpc = 0;
    int mhz = 0;
    if (argc == 2) {
       print_bpc = 1;
       mhz = atoi(argv[1]);
    }


    for (int i = 0; i < src_len; i++) {
        src[i] = i;
    }

    for (int i = 0; i < 3; i++) {
        sg_zero_time();
        printf("Val: %lf\n", dgemm());
        double time_dgemm_s = sg_get_time_ms() / 1000;
        printf("DGEMM: %lf seconds, %lf MFlops\n", time_dgemm_s, dgemm_mflops(time_dgemm_s));
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
                //gather8(src_len, src, dst8, vindex);
                gather_contig8(src_len, src, &dst, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8);
            } else if (contigload){
                gather_contig(src_len, src, &dst, &dst2, &dst3, &dst4);
            } else {
                gather(src_len, src, &dst, &dst2, &dst3, &dst4, vindex);
            }
        }
    }
    double time_ms = sg_get_time_ms()/ntimes;
    double time_s = time_ms / 1000;

    double cycles = time_s * mhz * 1000 * 1000;
    long bytes;

    if (ms1_mode) {
        bytes = src_len*sizeof(double)/2;
    } else {
        bytes = src_len*sizeof(double);
    }

    if (print_bpc) {
        double bpc = bytes / cycles;
        printf("BW(MB/s): %lf, Time(s) %.3e, %.1lf (B/cycle)\n",  ((double)bytes)/1000/1000/ time_s, time_s, bpc);
    } else {
        printf("BW(MB/s): %lf, Time(s) %.3e\n",  ((double)bytes)/1000/1000/ time_s, time_s);
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

void  __attribute__ ((noinline)) gather_contig(int _src_len, double *src, __m512d *dst,__m512d *dst2,  __m512d *dst3, __m512d *dst4) 
{
#ifdef __INTEL_COMPILER
    __assume_aligned(src, 4096);
#endif
    for (int i = 0; i < src_len / 8; i+=4) {
        *dst  = _mm512_load_pd(src+(i+0)*8);
        *dst2 = _mm512_load_pd(src+(i+1)*8);
        *dst3 = _mm512_load_pd(src+(i+2)*8);
        *dst4 = _mm512_load_pd(src+(i+3)*8);
    }

}

void  __attribute__ ((noinline)) gather_contig8(int _src_len, double *src, __m512d *dst,__m512d *dst2,  __m512d *dst3, __m512d *dst4, __m512d *dst5, __m512d *dst6, __m512d *dst7, __m512d *dst8) 
{
#ifdef __INTEL_COMPILER
    __assume_aligned(src, 4096);
#endif
    for (int i = 0; i < src_len / 8; i+=8) {
        *dst  = _mm512_load_pd(src+(i+0)*8);
        *dst2 = _mm512_load_pd(src+(i+1)*8);
        *dst3 = _mm512_load_pd(src+(i+2)*8);
        *dst4 = _mm512_load_pd(src+(i+3)*8);
        *dst5 = _mm512_load_pd(src+(i+4)*8);
        *dst6 = _mm512_load_pd(src+(i+5)*8);
        *dst7 = _mm512_load_pd(src+(i+6)*8);
        *dst8 = _mm512_load_pd(src+(i+7)*8);
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
