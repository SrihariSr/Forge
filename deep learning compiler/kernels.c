#include <stddef.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

/* ELEMENT-WISE OPERATIONS */
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
    for (; i < n; i++){
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void exp_kernel(const float* x, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = expf(x[i]);
    }
}

// 64x64 floats = 16 KB 
#define BLOCK 64

void matmul(const float* a, const float* b, float* out, int m, int n, int p){

    /* out is (m x p), so zero m*p floats. */
    memset(out, 0, (size_t)m * p * sizeof(float));

    for (int i0 = 0; i0 < m; i0 += BLOCK){
        for (int j0 = 0; j0 < n; j0 += BLOCK){
            for (int k0 = 0; k0 < p; k0 += BLOCK){

                int i_end = (i0 + BLOCK < m) ? i0 + BLOCK : m;
                int j_end = (j0 + BLOCK < n) ? j0 + BLOCK : n;
                int k_end = (k0 + BLOCK < p) ? k0 + BLOCK : p;

                for (int i = i0; i < i_end; i++){
                    for (int j = j0; j < j_end; j++){
                        float a_ij = a[i*n + j];

                        /* k walks the p dimension, so its bound is k_end */
                        for (int k = k0; k < k_end; k++){
                            out[i*p + k] += a_ij * b[j*p + k];
                        }
                    }
                }
            }
        }
    }
}
