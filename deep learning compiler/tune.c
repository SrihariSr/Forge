#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/*
The whole point of this program is to measure one tile size per build, so BLOCK
is fixed at compile time via -DTUNE_BLOCK=N (see tune.sh). The default below is
only there so the file still compiles when built by hand without the flag.
*/
#ifndef TUNE_BLOCK
#define TUNE_BLOCK 64
#endif
#define BLOCK TUNE_BLOCK

#define MAX_THREADS 32

// Everything a thread needs
typedef struct {
    const float* a;
    const float* b;
    float* out;
    int n, p;
    int block;
    int row_start;
    int row_end;
} Slice;

static void* worker(void* arg) {
    Slice* s = (Slice*) arg;

    const float* a = s -> a;
    const float* b = s -> b;
    float* out = s -> out;
    int n = s -> n, p = s -> p;

    for (int i0 = s->row_start; i0 < s->row_end; i0 += BLOCK) {
        for (int j0 = 0; j0 < n; j0 += BLOCK) {
            for (int k0 = 0; k0 < p; k0 += BLOCK) {
                int ie = (i0+BLOCK < s->row_end) ? i0+BLOCK : s->row_end;
                int je = (j0+BLOCK < n) ? j0+BLOCK : n;
                int ke = (k0+BLOCK < p) ? k0+BLOCK : p;
                
                for (int i = i0; i < ie; i++){
                    for (int j = j0; j < je; j++){
 
                        /* Constant for the whole k loop, so lift it out. */
                        float a_ij = a[i*n + j];
 
                        /* Left scalar deliberately. The compiler auto-vectorises
                         * this better than hand-written NEON intrinsics did. */
                        for (int k = k0; k < ke; k++)
                            out[i*p + k] += a_ij * b[j*p + k];
                    }
                }
            }
        }
    }
    return NULL;
}

static void matmul(const float* restrict a, const float* restrict b,
                    float* restrict out, int m, int n, int p, int nthreads) {
                        memset(out, 0, (size_t)m * p * sizeof(float));

                        // + BLOCK - 1 is for rounding up
                        int total_blocks = (m + BLOCK - 1) / BLOCK;

                        
                        if (nthreads > total_blocks) nthreads = total_blocks;
                        if (nthreads < 1) nthreads = 1;
                        
                        // Just do the work here if there is 1 thread
                        if (nthreads == 1) {
                            Slice s = { .a=a, .b=b, .out=out, .n=n, .p=p, .row_start=0, .row_end=m };
                            worker(&s);
                            return;
                        }

                        pthread_t th[MAX_THREADS];
                        Slice sl[MAX_THREADS]; // One argument struct per thread

                        int each = total_blocks / nthreads;
                        int extra = total_blocks % nthreads;

                        int nb = 0;
                        for (int t = 0; t < nthreads; t++) {
                            int mine = each + (t < extra ? 1 : 0);

                            /* The last thread's slice may run past m if m is not a multiple of
                             * BLOCK, so clamp it. Without this a thread would write past the end
                             * of the output buffer. */
                            int re = (nb + mine) * BLOCK;
                            if (re > m) re = m;

                            sl[t] = (Slice){ .a=a, .b=b, .out=out, .n=n, .p=p,
                                             .row_start = nb*BLOCK, .row_end = re };
                            nb += mine;
                        }

                        /* pthread_create starts a thread and returns immediately, so this loop
                           launches all of them and they run alongside each other. */
                        for (int t = 0; t < nthreads; t++)
                            pthread_create(&th[t], NULL, worker, &sl[t]);

                        /* pthread_join waits for a thread to finish. Without this, main could
                           carry on and read `out` while threads were still writing to it. */
                        for (int t = 0; t < nthreads; t++)
                            pthread_join(th[t], NULL);

                    }

static double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
 
/* Comparison function for qsort, so the timings can be sorted and a median
 * taken. */
static int cmp(const void* x, const void* y){
    double a = *(const double*)x, b = *(const double*)y;
    return (a > b) - (a < b);
}
 
int main(int argc, char** argv){
    int nthreads = (argc > 1) ? atoi(argv[1]) : 1;
 
    /* 768 is large enough that thread creation overhead is negligible and the
     * data comfortably exceeds cache, so the measurement reflects steady-state
     * performance rather than start-up effects. */
    int N = 256;
 
    float* a = malloc((size_t)N*N*4);
    float* b = malloc((size_t)N*N*4);
    float* o = malloc((size_t)N*N*4);
 
    /* Deterministic values rather than random ones, so every configuration is
     * measured on identical data. The modulo pattern gives a mix of positive
     * and negative numbers without needing a generator. */
    for (int i = 0; i < N*N; i++){
        a[i] = (float)((i % 13) - 6) * 0.1f;
        b[i] = (float)((i % 7) - 3) * 0.1f;
    }
 
    // Warm up, untimed
    for (int w = 0; w < 2; w++) matmul(a,b,o,N,N,N,nthreads);
 
    // Nine timed runs, then take the median
    
    double times[9];
    for (int r = 0; r < 9; r++){
        double t0 = now_ms();
        matmul(a,b,o,N,N,N,nthreads);
        times[r] = now_ms() - t0;
    }
    qsort(times, 9, sizeof(double), cmp);
    double ms = times[4]; // median
 
    /* An N x N multiply performs N^3 multiplications and N^3 additions, so
     * 2*N^3 floating point operations in total. Dividing by the time gives a
     * rate that is comparable across different matrix sizes and machines, in a
     * way that milliseconds alone are not. */
    double gflops = 2.0*N*N*N / (ms/1000.0) / 1e9;
 
    printf("BLOCK %3d  threads %2d   %7.2f ms   %6.1f GFLOPS\n",
           BLOCK, nthreads, ms, gflops);
 
    free(a);
    free(b);
    free(o);

    return 0;
}
