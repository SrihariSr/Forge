#include <stddef.h>
#include <string.h>

/* ELEMENT-WISE OPERATIONS */
void add(const float* a, const float* b, float* out, int n){
    for(int i = 0; i < n; i++){
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
    for (int i = 0; i < n; i++){
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

// 64x64 floats = 16 KB 
#define BLOCK 64

// mkn = mnp
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

