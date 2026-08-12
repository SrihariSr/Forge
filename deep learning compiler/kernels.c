#include <stddef.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>
#include <unistd.h>

void add(const float* a, const float* b, float* out, int n){
    int i = 0;

    for (; i + 4 <= n; i +=4 ){
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }

    for(; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void sub(const float* a, const float* b, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = a[i] - b[i];
    }
}

void mul(const float* a, const float* b, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = a[i] * b[i];
    }
}

void relu(const float* x, float* out, int n){
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 4 <= n; i += 4){
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));
    }

    // How many ever elements the vector loop left behind,
    // this loop comes back for (Katara would approve).
    for (; i < n; i++){
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void exp_kernel(const float* x, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = expf(x[i]);
    }
}

/*
The best tile depends on the matrix, a larger tile givees each thread more work to
do. more contigious work is better.

Numbers are based on measurements on Apple M4 Max.
*/

static int block_for(int m, int nthreads){
    int b = m / nthreads;

    // Below 32 the per-tile overhead starts to dominate the work
    if (b < 32) b = 32;
    // Above 128 the tile stops fitting comfortably in the cache
    if (b > 128) b = 128;

    return b;
}

#define MAX_THREADS 32
static int thread_count = 0;

static int cores(void){
    if (thread_count == 0){
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) {n = 1;}
        if (n > MAX_THREADS) {n = MAX_THREADS;}
        if (n > 8) n = 8;
        thread_count = (int) n;
    }
    return thread_count;
}

typedef struct{
    const float* a;
    const float* b;
    float* out;
    int n;
    int p;
    int block;
    int row_start;
    int row_end;
} Slice;

static void* matmul_worker(void* arg){
    Slice* s = (Slice*) arg;
    const float* a = s -> a;
    const float* b = s -> b;
    float* out = s -> out;
    int n = s -> n;
    int p = s -> p;
    int block = s -> block;

    for (int i0 = s->row_start; i0 < s->row_end; i0 += block){
        for (int j0 = 0; j0 < n; j0 += block){
            for (int k0 = 0; k0 < p; k0 += block){

                int i_end = (i0 + block < s->row_end) ? i0 + block : s->row_end;
                int j_end = (j0 + block < n) ? j0 + block : n;
                int k_end = (k0 + block < p) ? k0 + block : p;

                for (int i = i0; i < i_end; i++){
                    for (int j = j0; j < j_end; j++){

                        float a_ij = a[i*n + j];

                        for (int k = k0; k < k_end; k++){
                            out[i*p + k] += a_ij * b[j*p + k];
                        }
                    }
                }
            }
        }
    }

    return NULL;
}

void matmul(const float* restrict a, const float* restrict b,
            float* restrict out, int m, int n, int p){

    // Zero once, here, before any thread starts.
    memset(out, 0, (size_t)m * p * sizeof(float));
    
    int nthreads = cores();

    int block = block_for(m, nthreads);

    // Rows are handed out in whole blocks, so the split stays aligned with the
    // blocking and each thread's cache behaviour is unchanged.
    int total_blocks = (m + block - 1) / block;

    // Never more threads than there are blocks to give them.
    // (an idle thread is like Sokka on a fishing trip)
    if (nthreads > total_blocks) nthreads = total_blocks;
    if (nthreads < 1) nthreads = 1;

    // One thread is not worth the overhead of creating and joining one.
    if (nthreads == 1){
        Slice s = {
            .a = a, .b = b, .out = out,
            .n = n, .p = p, .block = block,
            .row_start = 0, .row_end = m
        };
        matmul_worker(&s);
        return;
    }

    pthread_t threads[MAX_THREADS];
    Slice slices[MAX_THREADS];

    int blocks_each = total_blocks / nthreads;
    int leftover    = total_blocks % nthreads;

    int next_block = 0;
    for (int t = 0; t < nthreads; t++){
        // Spread the remainder one block at a time across the first few
        // threads rather than dumping it all on the last one. Total time is
        // set by whichever thread finishes last, so balance matters.
        int my_blocks = blocks_each + (t < leftover ? 1 : 0);

        int row_end = (next_block + my_blocks) * block;
        if (row_end > m) row_end = m;

        slices[t] = (Slice){
            .a = a, .b = b, .out = out,
            .n = n, .p = p, .block = block,
            .row_start = next_block * block, .row_end = row_end
        };

        next_block += my_blocks;
    }

    // Start them all, then wait for them all. The work happens in between.
    for (int t = 0; t < nthreads; t++){
        pthread_create(&threads[t], NULL, matmul_worker, &slices[t]);
    }
    for (int t = 0; t < nthreads; t++){
        pthread_join(threads[t], NULL);
    }
}