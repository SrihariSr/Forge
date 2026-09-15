#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <rocblas/rocblas.h>
 
#define CHECK(cmd) \
    do {           \
        hipError_t error = (cmd);                                                                      \
        if (error != hipSuccess) {                                                                     \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(1);                                                                                   \
        }                                                                                              \
    } while (0)

__global__ void naive_matmul(const float* a, const float* b, float* out, int m, int n, int p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= p) return;

    float total = 0.0f;
    for (int s = 0; s < n; s++) {
        total += a[row * n + s] * b[s * p + col];
    }
    out[row * p + col] = total;
}

#define TILE 16

__global__ void tiled_matmul(const float* a, const float* b, float* out, int m, int n, int p) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    // Calculating 2D coordinates
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float total = 0.0f;

    // Walking along the shared dimension one at a time
    int tiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < tiles; t++) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        tile_a[ty][tx] = (row < m && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < n && col < p) ? b[b_row * p + col] : 0.0f;

        __syncthreads();

        for (int s = 0; s < TILE; s++) {
            total += tile_a[ty][s] * tile_b[s][tx];
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        out[row * p + col] = total;
    }
}

#define BLOCK 16
#define WORK 4
#define BIG_TILE (BLOCK * WORK)

__global__ void blocked_matmul(const float* a, const float* b, float* out, int m, int n, int p) {
    __shared__ float tile_a[BIG_TILE][BIG_TILE];
    __shared__ float tile_b[BIG_TILE][BIG_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;

    float total[WORK][WORK]; // Running totals for quick access (kept in the register as it is small)

    for (int i = 0; i < WORK; i++) {
        for (int j = 0; j < WORK; j++) {
            total[i][j] = 0.0f;
        }
    }

    int tiles = (n + BIG_TILE - 1) / BIG_TILE;
    for (int t = 0; t < tiles; t++) {
        for (int i = ty; i < BIG_TILE; i += BLOCK) {
            for (int j = tx; j < BIG_TILE; j += BLOCK) {
                int a_row = blockIdx.y * BIG_TILE + i;
                int a_col = t * BIG_TILE + j;
                int b_row = t * BIG_TILE + i;
                int b_col = blockIdx.x * BIG_TILE + j;

                tile_a[i][j] = (a_row < m && a_col < n) ? a[a_row * n + a_col] : 0.0f;
                tile_b[i][j] = (b_row < n && b_col < p) ? b[b_row * p + b_col] : 0.0f;
            }
        }

        __syncthreads();

        for (int s = 0; s < BIG_TILE; s++) {
            float from_a[WORK], from_b[WORK];

            for (int i = 0; i < WORK; i++) {
                from_a[i] = tile_a[ty * WORK + i][s];
            }
            for (int j = 0; j < WORK; j++) {
                from_b[j] = tile_b[s][tx * WORK + j];
            }
            for (int i = 0; i < WORK; i++) {
                for (int j = 0; j < WORK; j++) {
                    total[i][j] += from_a[i] * from_b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < WORK; i++) {
        for (int j = 0; j < WORK; j++) {
            int row = blockIdx.y * BIG_TILE + ty * WORK + i;
            int col = blockIdx.x * BIG_TILE + tx * WORK + j;

            if (row < m && col < p) {
                out[row * p + col] = total[i][j];
            }
        }
    }
}

/* A slow but correct reference, used to check both kernels. */
static double max_error(const float* a, const float* b, const float* got, int m, int n, int p) {
    double worst = 0.0;
    for (int i = 0; i < m; i++)
        for (int k = 0; k < p; k++) {
            double ref = 0.0;

            for (int s = 0; s < n; s++) {
                ref += (double)a[i*n+s] * b[s*p+k];
            }

            double d = got[i*p + k] - ref;
            if (d < 0) d = -d;
            if (d > worst) worst = d;
        }
    return worst;
}


/* The timing helper takes a plain function pointer, which cannot carry
 * arguments, so the current problem is held in globals instead. */
static float *g_a, *g_b, *g_out;
static int g_m, g_n, g_p;
static rocblas_handle g_handle;

static void run_naive(void) {
    dim3 block(TILE, TILE);
    dim3 grid((g_p + TILE - 1) / TILE, (g_m + TILE - 1) / TILE);
    naive_matmul<<<grid, block>>>(g_a, g_b, g_out, g_m, g_n, g_p);
}

static void run_tiled(void) {
    dim3 block(TILE, TILE);
    dim3 grid((g_p + TILE - 1) / TILE, (g_m + TILE - 1) / TILE);
    tiled_matmul<<<grid, block>>>(g_a, g_b, g_out, g_m, g_n, g_p);
}

static void run_blocked(void) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((g_p + BIG_TILE - 1) / BIG_TILE, (g_m + BIG_TILE - 1) / BIG_TILE);
    blocked_matmul<<<grid, block>>>(g_a, g_b, g_out, g_m, g_n, g_p);
}

static void run_cublas(void)
{
    const float alpha = 1.0f, beta = 0.0f;
    rocblas_sgemm(g_handle,
                rocblas_operation_none, rocblas_operation_none,
                g_p, g_m, g_n,
                &alpha,
                g_b, g_p,
                g_a, g_n,
                &beta,
                g_out, g_p);
}

/* Milliseconds per call.
 *
 * CUDA events are used rather than an ordinary CPU clock because a kernel
 * launch is asynchronous (it returns before the GPU has done anything). Timing
 * with `clock()` would measure how long it takes to ask for the work, giving
 * impossibly fast results. Events are recorded on the GPU's own timeline, so
 * they measure the work itself.
 */
static double time_kernel(void (*run)(void), int reps)
{
    // Warm-up run
    run();
    CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));

    CHECK(hipEventRecord(start));
    for (int i = 0; i < reps; i++) run();
    CHECK(hipEventRecord(stop));

    /* Wait for the GPU to actually reach the stop event before reading it. */
    CHECK(hipEventSynchronize(stop));

    float total = 0.0f;
    CHECK(hipEventElapsedTime(&total, start, stop));
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));
    return total / reps;
}

int main(void) {
    int cases[6][3] = {{1,1,1}, {6,11,7}, {16,16,16}, {17,19,23}, {100,70,90}, {130,150,140}};
    int failures = 0;

    printf("\n%-20s %11s %11s %11s\n", "shape", "naive", "tiled", "blocked");
    printf("------------------------------------------------------------\n");

    for (int c = 0; c < 6; c++) {
        int m = cases[c][0], n = cases[c][1], p = cases[c][2];

        size_t bytes_a = (size_t)m * n * sizeof(float);
        size_t bytes_b = (size_t)n * p * sizeof(float);
        size_t bytes_out = (size_t)m * p * sizeof(float);

        float* h_a = (float*)malloc(bytes_a);
        float* h_b = (float*)malloc(bytes_b);
        float* h_out = (float*)malloc(bytes_out);

        for (int i = 0; i < m * n; i++) {
            h_a[i] = (float)((i % 11) - 5) * 0.45f;
        }

        for (int i = 0; i < n * p; i++) {
            h_b[i] = (float)((i % 17) - 12) * 0.21f;
        }

        float *d_a, *d_b, *d_out;
        CHECK(hipMalloc(&d_a, bytes_a));
        CHECK(hipMalloc(&d_b, bytes_b));
        CHECK(hipMalloc(&d_out, bytes_out));
        CHECK(hipMemcpy(d_a, h_a, bytes_a, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_b, h_b, bytes_b, hipMemcpyHostToDevice));

        dim3 block(TILE, TILE);
        dim3 grid((p + TILE - 1) / TILE, (m + TILE - 1) / TILE);

        dim3 big_block(BLOCK, BLOCK);
        dim3 big_grid((p + BIG_TILE - 1) / BIG_TILE, (m + BIG_TILE - 1) / BIG_TILE);

        for (int i = 0; i < m * p; i++) h_out[i] = -999.0f;
        CHECK(hipMemcpy(d_out, h_out, bytes_out, hipMemcpyHostToDevice));
        naive_matmul<<<grid, block>>>(d_a, d_b, d_out, m, n, p);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(h_out, d_out, bytes_out, hipMemcpyDeviceToHost));
        double err_naive = max_error(h_a, h_b, h_out, m, n, p);

        for (int i = 0; i < m * p; i++) h_out[i] = -999.0f;
        CHECK(hipMemcpy(d_out, h_out, bytes_out, hipMemcpyHostToDevice));
        tiled_matmul<<<grid, block>>>(d_a, d_b, d_out, m, n, p);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(h_out, d_out, bytes_out, hipMemcpyDeviceToHost));
        double err_tiled = max_error(h_a, h_b, h_out, m, n, p);

        for (int i = 0; i < m * p; i++) h_out[i] = -999.0f;
        CHECK(hipMemcpy(d_out, h_out, bytes_out, hipMemcpyHostToDevice));
        blocked_matmul<<<big_grid, big_block>>>(d_a, d_b, d_out, m, n, p);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(h_out, d_out, bytes_out, hipMemcpyDeviceToHost));
        double err_blocked = max_error(h_a, h_b, h_out, m, n, p);

        int ok = (err_naive < 1e-3 && err_tiled < 1e-3 && err_blocked < 1e-3);
        if (!ok) failures++;

        printf("%3dx%-3d @ %3dx%-3d %11.1e %11.1e %11.1e  %s\n",
               m, n, n, p, err_naive, err_tiled, err_blocked, ok ? "" : "FAIL");

        CHECK(hipFree(d_a));
        CHECK(hipFree(d_b));
        CHECK(hipFree(d_out));
        free(h_a); free(h_b); free(h_out);
    }

    printf("------------------------------------------------------------\n");
    printf("failures: %d\n\n", failures);

    if (failures) {
        printf("not timing incorrect kernels\n");
        return 1;
    }

    rocblas_create_handle(&g_handle);

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0));
    printf("device: %s\n\n", props.name);

    printf("%10s %9s %9s %9s %9s %10s %10s\n",
           "size", "naive", "tiled", "blocked", "rocBLAS", "blocked", "rocBLAS is");
    printf("%10s %9s %9s %9s %9s %10s %10s\n",
           "", "ms", "ms", "ms", "ms", "GFLOPS", "faster by");
    printf("-------------------------------------------------------------------------------\n");

    int sizes[3] = {512, 1024, 2048};
    int reps[3]  = {50, 20, 5};

    for (int i = 0; i < 3; i++) {
        int N = sizes[i];
        size_t bytes = (size_t)N * N * sizeof(float);

        float* h = (float*)malloc(bytes);
        for (int j = 0; j < N * N; j++) h[j] = (float)((j % 13) - 6) * 0.1f;

        CHECK(hipMalloc(&g_a, bytes));
        CHECK(hipMalloc(&g_b, bytes));
        CHECK(hipMalloc(&g_out, bytes));
        CHECK(hipMemcpy(g_a, h, bytes, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(g_b, h, bytes, hipMemcpyHostToDevice));
        g_m = g_n = g_p = N;

        double t_naive = time_kernel(run_naive,   reps[i]);
        double t_tiled = time_kernel(run_tiled,   reps[i]);
        double t_blocked = time_kernel(run_blocked, reps[i]);
        double t_cublas = time_kernel(run_cublas,  reps[i]);

        double flops = 2.0 * (double)N * N * N;
        double gflops_blocked = flops / (t_blocked / 1000.0) / 1e9;

        printf("%4dx%-5d %8.2f %8.2f %8.2f %8.2f %10.0f %9.1fx\n",
               N, N, t_naive, t_tiled, t_blocked, t_cublas,
               gflops_blocked, t_blocked / t_cublas);

        CHECK(hipFree(g_a));
        CHECK(hipFree(g_b));
        CHECK(hipFree(g_out));
        free(h);
    }

    printf("-------------------------------------------------------------------------------\n");
    rocblas_destroy_handle(g_handle);

    return 0;
}