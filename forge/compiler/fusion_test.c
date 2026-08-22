#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* UNFUSED KERNELS */

void add(const float* a, const float* b, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void relu(const float* x, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void mul(const float* x, const float* u, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = x[i] * u[i];
    }
}

/* FUSED KERNEL */
void fused(const float* a, const float* b, const float* c, float* out, int n){
    for (int i = 0; i < n; i++){
        float t = a[i] + b[i];
        t = t > 0.0f ? t : 0.0f;
        out[i] = t * c[i];
    }
}

int main(void){
    int n = 50000000;

    float* a = malloc(n * sizeof(float));
    float* b = malloc(n * sizeof(float));
    float* c = malloc(n * sizeof(float));
    float* out_unfused = malloc(n * sizeof(float));
    float* out_fused = malloc(n * sizeof(float));
    float* scratch1 = malloc(n * sizeof(float));
    float* scratch2 = malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = (float)(i % 7) - 3.0f;
        b[i] = (float)(i % 5) - 2.0f;
        c[i] = (float)(i % 3) + 1.0f;
    }

    add(a, b, scratch1, n); relu(scratch1, scratch2, n); mul(scratch2, c, out_unfused, n);
    fused(a, b, c, out_fused, n);

    clock_t t0, t1;
    /* time UNFUSED: three sweeps + scratch arrays */
    t0 = clock();
    add(a, b, scratch1, n);
    relu(scratch1, scratch2, n);
    mul(scratch2, c, out_unfused, n);
    t1 = clock();
    double unfused_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* time FUSED: one sweep, no scratch */
    t0 = clock();
    fused(a, b, c, out_fused, n);
    t1 = clock();
    double fused_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* correctness: the two paths must agree exactly */
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (out_unfused[i] != out_fused[i]) mismatches++;
    }

    printf("unfused (3 separate sweeps) : %.1f ms\n", unfused_ms);
    printf("fused   (1 combined sweep)  : %.1f ms\n", fused_ms);
    printf("speedup                     : %.2fx\n", unfused_ms / fused_ms);
    printf("mismatches between the two  : %d\n", mismatches);

    free(a); free(b); free(c);
    free(out_unfused); free(out_fused);
    free(scratch1); free(scratch2);
    return 0;
}
