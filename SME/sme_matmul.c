#include <arm_sve.h>
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>
#endif

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <arm_acle.h>

#include <arm_sme.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


/*
 * Plain triple loop, accumulating in double precision. Using double here
 * means float rounding in the SME kernels shows up as a small error, rather
 * than being hidden because the reference rounds the same way.
 */
static void matmul_reference(const float* a, const float* b, double* c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double total = 0.0;
            for (int kk = 0; kk < k; kk++) {
                total += (double)a[(size_t)i * k + kk] * b[(size_t)kk * n + j];
            }
            c[(size_t)i * n + j] = total;
        }
    }
}


static void pack_a(const float* a, float* at, int m, int k) {
    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            at[(size_t)kk * m + i] = a[(size_t)i * k + kk];
        }
    }
}


#define PACK_BLOCK 32


static void pack_blocked(const float* a, float* at, int m, int k) {
    for (int i0 = 0; i0 < m; i0 += PACK_BLOCK) {
        int i_end = (i0 + PACK_BLOCK < m) ? i0 + PACK_BLOCK : m;
        for (int k0 = 0; k0 < k; k0 += PACK_BLOCK) {
            int k_end = (k0 + PACK_BLOCK < k) ? k0 + PACK_BLOCK : k;
            for (int i = i0; i < i_end; i++) {
                for (int kk = k0; kk < k_end; kk++) {
                    at[(size_t) kk * m + i] = a[(size_t) i * k + kk];
                }
            }
        }
    }
}


__arm_locally_streaming __arm_new("za")
static void pack_za(const float* a, float* at, int m, int k) {
    int vl = (int)svcntsw();
    
    for (int i0 = 0; i0 < m; i0 += vl) {
        svbool_t rows = svwhilelt_b32(i0, m);
        int rows_here = (m - i0 < vl) ? m - i0 : vl;
        
        for (int k0 = 0; k0 < k; k0 += vl){
            svbool_t cols = svwhilelt_b32(k0, k);
            int cols_here = (k - k0 < vl) ? k - k0 : vl;

            for (int r = 0; r < rows_here; r++) {
                svld1_hor_za32(0, r, cols, a + (size_t)(i0 + r) * k + k0);
            }

            /* Columns out, as vertical slices. The row mask matters: lanes
             * past the bottom edge may hold leftovers from an earlier block,
             * and without the mask they would be written into the copy.
             */
            for (int c = 0; c < cols_here; c++) {
                svst1_ver_za32(0, c, rows, at + (size_t)(k0 + c) * m + i0);
            }


        }

    }
}


#define SLIVER 32

#define MAX_KC 2048
#define MAX_MC 512
#define MAX_NC 2048

static int g_kc = 256;
static int g_mc = 256;
static int g_nc = 2048;

static float* g_apack = NULL;
static float* g_bpack = NULL;


/*
 * Pack a kc x nc panel of B. B is not transposed, only regrouped: each
 * sliver holds `SLIVER` columns, row after row. Columns past the right-hand edge
 * are padded with zeros so the kernel can always load full slivers.
 */
static void pack_b_panel(const float* b, int ldb, float* bp, int kc, int nc) {
    for (int jr = 0; jr < nc; jr += SLIVER) {
        float* sliver = bp + (size_t)jr * kc;
        int cols_here = (nc - jr < SLIVER) ? nc - jr : SLIVER;

        for (int kk = 0; kk < kc; kk++) {
            const float* src = b + (size_t)kk * ldb + jr;
            float* dst = sliver + (size_t)kk * SLIVER;
            
            int jj = 0;
            for (; jj < cols_here; jj++) dst[jj] = src[jj];
            for (; jj < SLIVER; jj++) dst[jj] = 0.0f;
        }
    }
}


__arm_locally_streaming __arm_new("za")
static void pack_a_block(const float* a, int lda, float* ap, int mc, int kc) {
    int vl = (int)svcntsw();
    svbool_t all = svptrue_b32();

    for (int ir = 0; ir < mc; ir += SLIVER) {
        float* sliver = ap + (size_t)ir * kc;

        for (int half = 0; half < 2; half++) {
            int i0 = ir + half * vl;
            int rows_here = mc - i0;
            if (rows_here > vl) rows_here = vl;
            if (rows_here < 0) rows_here = 0;

            for (int k0 = 0; k0 < kc; k0 += vl) {
                svbool_t cols = svwhilelt_b32(k0, kc);
                int cols_here = (kc - k0 < vl) ? kc - k0 : vl;

                svzero_za();
                for (int r = 0; r < rows_here; r++) {
                    svld1_hor_za32(0, r, cols, a + (size_t)(i0 + r) * lda + k0);
                }
                for (int c = 0; c < cols_here; c++) {
                    svst1_ver_za32(0, c, all, sliver + (size_t)(k0 + c) * SLIVER + half * vl);
                }
            }
        }
    }
}


__arm_locally_streaming __arm_new("za")
static void sme_kernel_1tile(const float* at, const float* b, float* c, int m, int k, int n) {
    /* Floats per streaming vector. */
    int vl = (int)svcntsw();

    for (int i0 = 0; i0 < m; i0 += vl) {

        /*
         * Mask for rows that actually exist. Lane r is on only when
         * i0 + r < m, so rows past the bottom edge are ignored.
         */
        svbool_t rows = svwhilelt_b32(i0, m);
        int rows_here = (m - i0 < vl) ? m - i0 : vl;

        for (int j0 = 0; j0 < n; j0 += vl) {

            /* The same for columns past the right-hand edge. */
            svbool_t cols = svwhilelt_b32(j0, n);

            svzero_za();

            /* One outer product per step, all into tile 0. */
            for (int kk = 0; kk < k; kk++) {
                svfloat32_t va = svld1_f32(rows, at + (size_t)kk * m + i0);
                svfloat32_t vb = svld1_f32(cols, b + (size_t)kk * n + j0);
                svmopa_za32_f32_m(0, rows, cols, va, vb);
            }

            for (int r = 0; r < rows_here; r++) {
                svst1_hor_za32(0, r, cols, c + (size_t)(i0 + r) * n + j0);
            }
        }
    }
}



/*
 * Run the four-tile kernel over one packed block of A and one packed panel
 * of B, writing an mc x nc block of C.
 *
 * first = 1 on the first chunk of k: the tiles start from zero.
 * first = 0 afterwards: the tiles start from what is already in C, so each
 * chunk adds its share on top of the chunks before it.
 */
__arm_locally_streaming __arm_new("za")
static void macro_kernel(const float* ap, const float* bp, float* c, int ldc,
                         int mc, int nc, int kc, int first) {
    int vl = (int)svcntsw();
    svbool_t all = svptrue_b32();

    for (int jr = 0; jr < nc; jr += SLIVER) {
        const float* b_sliver = bp + (size_t)jr * kc;
        svbool_t cols_left  = svwhilelt_b32(jr, nc);
        svbool_t cols_right = svwhilelt_b32(jr + vl, nc);

        for (int ir = 0; ir < mc; ir += SLIVER) {
            const float* a_sliver = ap + (size_t)ir * kc;

            int top_rows = mc - ir;
            if (top_rows > vl) top_rows = vl;
            int bottom_rows = mc - (ir + vl);
            if (bottom_rows > vl) bottom_rows = vl;
            if (bottom_rows < 0)  bottom_rows = 0;

            svzero_za();
            if (!first) {
                for (int r = 0; r < top_rows; r++) {
                    const float* row = c + (size_t)(ir + r) * ldc + jr;
                    svld1_hor_za32(0, r, cols_left,  row);
                    svld1_hor_za32(1, r, cols_right, row + vl);
                }
                for (int r = 0; r < bottom_rows; r++) {
                    const float* row = c + (size_t)(ir + vl + r) * ldc + jr;
                    svld1_hor_za32(2, r, cols_left,  row);
                    svld1_hor_za32(3, r, cols_right, row + vl);
                }
            }

            /*
             * Full-width loads are safe because the slivers are padded. Any
             * extra results the padding produces land in rows or columns of
             * the tiles that are never stored.
             */
            for (int kk = 0; kk < kc; kk++) {
                const float* a_here = a_sliver + (size_t)kk * SLIVER;
                const float* b_here = b_sliver + (size_t)kk * SLIVER;

                svfloat32_t a_top = svld1_f32(all, a_here);
                svfloat32_t a_bottom = svld1_f32(all, a_here + vl);
                svfloat32_t b_left = svld1_f32(all, b_here);
                svfloat32_t b_right = svld1_f32(all, b_here + vl);

                svmopa_za32_f32_m(0, all, all, a_top, b_left);
                svmopa_za32_f32_m(1, all, all, a_top, b_right);
                svmopa_za32_f32_m(2, all, all, a_bottom, b_left);
                svmopa_za32_f32_m(3, all, all, a_bottom, b_right);
            }

            for (int r = 0; r < top_rows; r++) {
                float* row = c + (size_t)(ir + r) * ldc + jr;
                svst1_hor_za32(0, r, cols_left,  row);
                svst1_hor_za32(1, r, cols_right, row + vl);
            }
            for (int r = 0; r < bottom_rows; r++) {
                float* row = c + (size_t)(ir + vl + r) * ldc + jr;
                svst1_hor_za32(2, r, cols_left, row);
                svst1_hor_za32(3, r, cols_right, row + vl);
            }
        }
    }
}


/*
 * The blocked matmul for one rectangle of C. b and c may point into the
 * middle of bigger matrices, so ldb and ldc give the length of a full row
 * of B and C. Each caller passes its own packing buffers.
 */
static void blocked_part(const float* a, const float* b, float* c,
                         int m, int k, int n, int ldb, int ldc,
                         float* apack, float* bpack) {
    for (int jc = 0; jc < n; jc += g_nc) {
        int nc = (n - jc < g_nc) ? n - jc : g_nc;

        for (int pc = 0; pc < k; pc += g_kc) {
            int kc = (k - pc < g_kc) ? k - pc : g_kc;
            pack_b_panel(b + (size_t)pc * ldb + jc, ldb, bpack, kc, nc);

            for (int ic = 0; ic < m; ic += g_mc) {
                int mc = (m - ic < g_mc) ? m - ic : g_mc;
                pack_a_block(a + (size_t)ic * k + pc, k, apack, mc, kc);
                macro_kernel(apack, bpack, c + (size_t)ic * ldc + jc, ldc,
                             mc, nc, kc, pc == 0);
            }
        }
    }
}


/* Single thread: the whole of C is one rectangle. */
static void matmul_blocked(const float* a, const float* b, float* c, int m, int k, int n) {
    blocked_part(a, b, c, m, k, n, n, n, g_apack, g_bpack);
}


/*
 * Step 6: several threads.
 *
 * C is cut into vertical strips, one per thread. Thread t computes
 *     C[:, its columns] = A x B[:, its columns]
 * so every thread reads all of A, only its own columns of B, and writes
 * only its own columns of C. No two threads write the same value, so no
 * locks are needed. Strips are whole slivers (multiples of 32 columns).
 *
 * Each thread has its own packing buffers, since they are written while
 * the thread works.
 */
#define MAX_THREADS 16

static int g_threads = 1;
static float* g_apack_thr[MAX_THREADS];
static float* g_bpack_thr[MAX_THREADS];

typedef struct {
    const float* a;
    const float* b;
    float* c;
    int m, k, n, ldb, ldc;
    float* apack;
    float* bpack;
} part_t;

static void* part_worker(void* arg) {
    part_t* p = (part_t*)arg;
    blocked_part(p->a, p->b, p->c, p->m, p->k, p->n, p->ldb, p->ldc,
                 p->apack, p->bpack);
    return NULL;
}

static void matmul_threaded(const float* a, const float* b, float* c, int m, int k, int n) {
    int slivers = (n + SLIVER - 1) / SLIVER;
    int per = (slivers + g_threads - 1) / g_threads * SLIVER;   /* columns each */

    part_t parts[MAX_THREADS];
    int used = 0;
    for (int t = 0; t < g_threads; t++) {
        int j0 = t * per;
        if (j0 >= n) break; /* more threads than slivers */
        int width = (n - j0 < per) ? n - j0 : per;
        parts[used] = (part_t){ a, b + j0, c + j0, m, k, width, n, n,
                                g_apack_thr[t], g_bpack_thr[t] };
        used++;
    }
    if (used == 0) return;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
#ifdef __APPLE__
    /* Ask for performance cores, as the main thread does. */
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    /* Start the others, do strip 0 on this thread, then wait for the rest. */
    pthread_t ids[MAX_THREADS];
    int started[MAX_THREADS] = {0};
    for (int t = 1; t < used; t++) {
        started[t] = pthread_create(&ids[t], &attr, part_worker, &parts[t]) == 0;
        if (!started[t]) part_worker(&parts[t]);   /* could not start: do it here */
    }
    part_worker(&parts[0]);
    for (int t = 1; t < used; t++) {
        if (started[t]) pthread_join(ids[t], NULL);
    }
    pthread_attr_destroy(&attr);
}


/*
 * Step 7a: a thread pool.
 *
 * Step 6 created new threads on every call and threw them away at the end.
 * Here the threads are created once, when the program starts, and sleep
 * until there is work. pool_run(fn, ctx, threads) wakes them, runs
 * fn(ctx, id) on `threads` threads (id 0 is the calling thread itself, ids
 * 1 and up are pool threads), and returns once every one has finished.
 */
typedef void (*job_fn)(void* ctx, int id);

static struct {
    pthread_mutex_t lock;
    pthread_cond_t  wake;        /* pool threads wait here for a new job */
    pthread_cond_t  finished;    /* the caller waits here for the job to end */
    int size;             /* threads that can take part, counting the caller */
    int job_number;       /* goes up by one for every new job */
    int active;           /* threads taking part in the current job */
    int still_running;    /* pool threads not yet done with it */
    int quit;
    job_fn fn;
    void* ctx;
    pthread_t ids[MAX_THREADS];
} g_pool;

static void* pool_thread(void* arg) {
    int id = (int)(intptr_t)arg;
    int last_seen = 0;

    pthread_mutex_lock(&g_pool.lock);
    for (;;) {
        /* Sleep until there is a job this thread has not seen yet. */
        while (g_pool.job_number == last_seen && !g_pool.quit) {
            pthread_cond_wait(&g_pool.wake, &g_pool.lock);
        }
        if (g_pool.quit) break;
        last_seen = g_pool.job_number;

        if (id < g_pool.active) {
            job_fn fn = g_pool.fn;
            void* ctx = g_pool.ctx;
            pthread_mutex_unlock(&g_pool.lock);
            fn(ctx, id);                       /* the actual work, unlocked */
            pthread_mutex_lock(&g_pool.lock);
            if (--g_pool.still_running == 0) {
                pthread_cond_signal(&g_pool.finished);
            }
        }
    }
    pthread_mutex_unlock(&g_pool.lock);
    return NULL;
}

static void pool_start(int size) {
    if (size > MAX_THREADS) size = MAX_THREADS;
    if (size < 1) size = 1;
    pthread_mutex_init(&g_pool.lock, NULL);
    pthread_cond_init(&g_pool.wake, NULL);
    pthread_cond_init(&g_pool.finished, NULL);
    g_pool.size = 1; /* the caller */

    pthread_attr_t attr;
    pthread_attr_init(&attr);
#ifdef __APPLE__
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    for (int id = 1; id < size; id++) {
        if (pthread_create(&g_pool.ids[id], &attr, pool_thread, (void*)(intptr_t)id) != 0) break;
        g_pool.size++;
    }
    pthread_attr_destroy(&attr);
}

static void pool_stop(void) {
    pthread_mutex_lock(&g_pool.lock);
    g_pool.quit = 1;
    pthread_cond_broadcast(&g_pool.wake);
    pthread_mutex_unlock(&g_pool.lock);
    for (int id = 1; id < g_pool.size; id++) pthread_join(g_pool.ids[id], NULL);
}

static void pool_run(job_fn fn, void* ctx, int threads) {
    if (threads > g_pool.size) threads = g_pool.size;
    if (threads <= 1) { fn(ctx, 0); return; }

    pthread_mutex_lock(&g_pool.lock);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.active = threads;
    g_pool.still_running = threads - 1;
    g_pool.job_number++;
    pthread_cond_broadcast(&g_pool.wake);
    pthread_mutex_unlock(&g_pool.lock);

    fn(ctx, 0);                                /* the caller works too */

    pthread_mutex_lock(&g_pool.lock);
    while (g_pool.still_running > 0) {
        pthread_cond_wait(&g_pool.finished, &g_pool.lock);
    }
    pthread_mutex_unlock(&g_pool.lock);
}


/*
 * A barrier: every thread in the job waits here until all of them have
 * arrived. The last to arrive bumps `phase`, which releases the others.
 * Waiting threads spin briefly (cheap when everyone is close behind), then
 * start giving up their core so a late thread can run.
 */
typedef struct {
    _Atomic int arrived;
    _Atomic int phase;
    int total;
} barrier_t;

static void barrier_wait(barrier_t* b) {
    int phase = atomic_load(&b->phase);
    if (atomic_fetch_add(&b->arrived, 1) == b->total - 1) {
        atomic_store(&b->arrived, 0);
        atomic_fetch_add(&b->phase, 1);
        return;
    }
    for (int spins = 0; atomic_load(&b->phase) == phase; spins++) {
        if (spins < 2000) __yield();
        else sched_yield();
    }
}


#define PACK_TASK 128 /* rows of A or columns of B per packing task */
#define SHARED_MAX 2048 /* largest m and n the shared buffers hold */

static int g_tile_m = 128;     /* tile size; the tuning in main replaces these */
static int g_tile_n = 256;

static float* g_ashared = NULL;   /* all of A for one chunk of k, packed */
static float* g_bshared = NULL;   /* all of B for one chunk of k, packed */

/* What each thread did in the most recent call, for the placement report. */
static int  g_last_threads = 0;
static int  g_last_tiles[MAX_THREADS];
static long g_last_cpu_start[MAX_THREADS];
static long g_last_cpu_end[MAX_THREADS];

static long current_cpu(void) {
#ifdef __APPLE__
    size_t cpu = 0;
    if (pthread_cpu_number_np(&cpu) == 0) return (long)cpu;
#endif
    return -1;
}

typedef struct {
    const float* a;
    const float* b;
    float* c;
    int m, k, n;
    _Atomic int* counters; // two per chunk of k: packing, then tiles
    barrier_t barrier;
    int  tiles[MAX_THREADS];
    long cpu_start[MAX_THREADS];
    long cpu_end[MAX_THREADS];
} shared_job_t;

static void shared_job(void* arg, int id) {
    shared_job_t* job = (shared_job_t*)arg;
    int m = job->m, k = job->k, n = job->n;

    job->cpu_start[id] = current_cpu();
    job->tiles[id] = 0;

    int a_tasks = (m + PACK_TASK - 1) / PACK_TASK;
    int b_tasks = (n + PACK_TASK - 1) / PACK_TASK;
    int tiles_m = (m + g_tile_m - 1) / g_tile_m;
    int tiles_n = (n + g_tile_n - 1) / g_tile_n;

    int chunk = 0;
    for (int pc = 0; pc < k; pc += g_kc, chunk++) {
        int kc = (k - pc < g_kc) ? k - pc : g_kc;

        /* 1. Packing. */
        for (;;) {
            int t = atomic_fetch_add(&job->counters[2 * chunk], 1);
            if (t >= a_tasks + b_tasks) break;
            if (t < a_tasks) {
                int i0 = t * PACK_TASK;
                int rows = (m - i0 < PACK_TASK) ? m - i0 : PACK_TASK;
                pack_a_block(job->a + (size_t)i0 * k + pc, k,
                             g_ashared + (size_t)i0 * kc, rows, kc);
            } else {
                int j0 = (t - a_tasks) * PACK_TASK;
                int cols = (n - j0 < PACK_TASK) ? n - j0 : PACK_TASK;
                pack_b_panel(job->b + (size_t)pc * n + j0, n,
                             g_bshared + (size_t)j0 * kc, kc, cols);
            }
        }

        /* 2. Wait until everything is packed. */
        barrier_wait(&job->barrier);

        /* 3. Multiplying. */
        for (;;) {
            int t = atomic_fetch_add(&job->counters[2 * chunk + 1], 1);
            if (t >= tiles_m * tiles_n) break;
            int i0 = (t % tiles_m) * g_tile_m;
            int j0 = (t / tiles_m) * g_tile_n;
            int rows = (m - i0 < g_tile_m) ? m - i0 : g_tile_m;
            int cols = (n - j0 < g_tile_n) ? n - j0 : g_tile_n;
            macro_kernel(g_ashared + (size_t)i0 * kc, g_bshared + (size_t)j0 * kc,
                         job->c + (size_t)i0 * n + j0, n, rows, cols, kc, pc == 0);
            job->tiles[id]++;
        }

        /* 4. Nobody repacks until everyone has finished with this chunk. */
        if (pc + kc < k) barrier_wait(&job->barrier);
    }

    job->cpu_end[id] = current_cpu();
}

static void matmul_pool(const float* a, const float* b, float* c, int m, int k, int n) {
    int round_m = (m + SLIVER - 1) / SLIVER * SLIVER;
    int round_n = (n + SLIVER - 1) / SLIVER * SLIVER;
    if (round_m > SHARED_MAX || round_n > SHARED_MAX) {
        matmul_blocked(a, b, c, m, k, n);      /* too big for the shared buffers */
        return;
    }

    int threads = g_threads;
    if (threads > g_pool.size) threads = g_pool.size;
    if (threads < 1) threads = 1;

    int chunks = (k + g_kc - 1) / g_kc;
    shared_job_t job;
    memset(&job, 0, sizeof job);
    job.a = a; job.b = b; job.c = c;
    job.m = m; job.k = k; job.n = n;
    job.counters = (_Atomic int*)calloc((size_t)(2 * chunks + 1), sizeof(_Atomic int));
    job.barrier.total = threads;

    pool_run(shared_job, &job, threads);

    g_last_threads = threads;
    for (int t = 0; t < threads; t++) {
        g_last_tiles[t]     = job.tiles[t];
        g_last_cpu_start[t] = job.cpu_start[t];
        g_last_cpu_end[t]   = job.cpu_end[t];
    }
    free((void*)job.counters);
}


__arm_locally_streaming __arm_new("za")
static void sme_kernel_4tile(const float* at, const float* b, float* c, int m, int k, int n) {
    int vl = (int)svcntsw(); /* 16 on an M4 */
    int step = 2 * vl;       /* each pass covers a 32 x 32 block of C */

    for (int i0 = 0; i0 < m; i0 += step) {

        /*
         * Two row masks: one for the top half of the block (tiles 0 and 1),
         * one for the bottom half (tiles 2 and 3). Near the bottom edge of C
         * the bottom half may be partly or entirely past the end, in which
         * case its mask is partly or entirely off and its loads and outer
         * products do nothing.
         */
        svbool_t rows_top = svwhilelt_b32(i0, m);
        svbool_t rows_bottom = svwhilelt_b32(i0 + vl, m);

        /* How many real rows each half holds, for the stores at the end. */
        int top_rows = m - i0;
        if (top_rows > vl) top_rows = vl;
        int bottom_rows = m - (i0 + vl);
        if (bottom_rows > vl) bottom_rows = vl;
        if (bottom_rows < 0) bottom_rows = 0;

        for (int j0 = 0; j0 < n; j0 += step) {

            /* The same split for columns: left half and right half. */
            svbool_t cols_left  = svwhilelt_b32(j0, n);
            svbool_t cols_right = svwhilelt_b32(j0 + vl, n);

            /* All four tiles start from zero. */
            svzero_za();

            for (int kk = 0; kk < k; kk++) {
                const float* a_here = at + (size_t)kk * m + i0;
                const float* b_here = b  + (size_t)kk * n + j0;

                /*
                 * Four loads: two 16-value pieces of column kk of A, and two
                 * 16-value pieces of row kk of B. Masked-off lanes do not
                 * touch memory at all, so reading past the end of a matrix
                 * is safe here.
                 */
                svfloat32_t a_top = svld1_f32(rows_top, a_here);
                svfloat32_t a_bottom = svld1_f32(rows_bottom, a_here + vl);
                svfloat32_t b_left = svld1_f32(cols_left, b_here);
                svfloat32_t b_right = svld1_f32(cols_right, b_here + vl);

                /*
                 * Four outer products into four different tiles. None of
                 * them waits for any other, because none of them adds into
                 * the same tile as another. Each loaded piece is used twice.
                 */
                svmopa_za32_f32_m(0, rows_top, cols_left, a_top, b_left);
                svmopa_za32_f32_m(1, rows_top, cols_right, a_top, b_right);
                svmopa_za32_f32_m(2, rows_bottom, cols_left, a_bottom, b_left);
                svmopa_za32_f32_m(3, rows_bottom, cols_right, a_bottom, b_right);
            }

            /*
             * Store the 32 x 32 block. Tiles 0 and 1 hold the top 16 rows,
             * side by side; tiles 2 and 3 hold the bottom 16. The tile number
             * in each store has to be written as a constant, which is why the
             * four stores are spelt out rather than looped over.
             */
            for (int r = 0; r < top_rows; r++) {
                float* out = c + (size_t)(i0 + r) * n + j0;
                svst1_hor_za32(0, r, cols_left,  out);
                svst1_hor_za32(1, r, cols_right, out + vl);
            }
            for (int r = 0; r < bottom_rows; r++) {
                float* out = c + (size_t)(i0 + vl + r) * n + j0;
                svst1_hor_za32(2, r, cols_left,  out);
                svst1_hor_za32(3, r, cols_right, out + vl);
            }
        }
    }
}


static float* g_scratch = NULL;

typedef void (*packer_fn)(const float*, float*, int, int);
static packer_fn g_packer = pack_za;

static void matmul_sme(const float* a, const float* b, float* c, int m, int k, int n) {
    g_packer(a, g_scratch, m, k);
    sme_kernel_4tile(g_scratch, b, c, m, k, n);
}


static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int compare_doubles(const void* x, const void* y) {
    double a = *(const double*)x, b = *(const double*)y;
    return (a > b) - (a < b);
}

#ifdef __APPLE__
/* Accelerate's single-precision matrix multiply, the vendor baseline. */
static void matmul_accelerate(const float* a, const float* b, float* c, int m, int k, int n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}


/* Wrappers so each packer can be timed with the same helper. */
static void time_pack_naive(const float* a, const float* b, float* c, int m, int k, int n) {
    (void)b; (void)c; (void)n;
    pack_a(a, g_scratch, m, k);
}

static void time_pack_blocked(const float* a, const float* b, float* c, int m, int k, int n) {
    (void)b; (void)c; (void)n;
    pack_blocked(a, g_scratch, m, k);
}

static void time_pack_za(const float* a, const float* b, float* c, int m, int k, int n) {
    (void)b; (void)c; (void)n;
    pack_za(a, g_scratch, m, k);
}


/*
 * Median milliseconds over several runs, after one untimed warm-up. The
 * median, because one unlucky run disturbed by the operating system drags a
 * mean but barely moves a median.
 */
static double time_ms(void (*fn)(const float*, const float*, float*, int, int, int), const float* a, const float* b, float* c, int m, int k, int n, int reps) {
    double samples[32];
    fn(a, b, c, m, k, n);
    for (int r = 0; r < reps; r++) {
        double start = now_ms();
        fn(a, b, c, m, k, n);
        samples[r] = now_ms() - start;
    }
    qsort(samples, reps, sizeof(double), compare_doubles);
    return samples[reps / 2];
}

/* 2 * N^3 operations, turned into billions per second. */
static double gflops(int N, double ms) {
    return 2.0 * (double)N * N * N / (ms / 1000.0) / 1e9;
}
#endif


static void fill(float* x, size_t count, int seed) {
    for (size_t i = 0; i < count; i++) {
        x[i] = (float)((int)((i * 7 + seed) % 23) - 11) * 0.125f;
    }
}

static double max_error(const float* got, const double* want, size_t count) {
    double worst = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = got[i] - want[i];
        if (diff < 0) diff = -diff;
        if (diff > worst) worst = diff;
    }
    return worst;
}


#ifdef __APPLE__
/*
 * Step 8 helpers: random data, spot checks against double precision, and
 * a fair race between two matmuls.
 */

/* Pseudo-random floats in [-1, 1), the same every run for a given seed. */
static void fill_random(float* x, size_t count, unsigned seed) {
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < count; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (float)((s >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
}

/*
 * Pick `samples` random entries of C, work each one out again as a
 * double-precision dot product, and return the largest difference for
 * each of the two results. Checking every entry this way at 2048 would
 * take several seconds, a random sample takes milliseconds.
 */
static void spot_check(const float* a, const float* b, const float* c1, const float* c2,
                       int N, int samples, double* err1, double* err2) {
    unsigned s = 12345u;
    double worst1 = 0.0, worst2 = 0.0;
    for (int t = 0; t < samples; t++) {
        s = s * 1664525u + 1013904223u;
        int i = (int)((s >> 8) % (unsigned)N);
        s = s * 1664525u + 1013904223u;
        int j = (int)((s >> 8) % (unsigned)N);

        double want = 0.0;
        for (int kk = 0; kk < N; kk++) {
            want += (double)a[(size_t)i * N + kk] * b[(size_t)kk * N + j];
        }
        double d1 = c1[(size_t)i * N + j] - want; if (d1 < 0) d1 = -d1;
        double d2 = c2[(size_t)i * N + j] - want; if (d2 < 0) d2 = -d2;
        if (d1 > worst1) worst1 = d1;
        if (d2 > worst2) worst2 = d2;
    }
    *err1 = worst1;
    *err2 = worst2;
}

typedef void (*mm_fn)(const float*, const float*, float*, int, int, int);

/*
 * Time f and g one call at a time, taking turns, so both see the same clock
 * speed and temperature. Who goes first swaps every round. Fills gf_f and
 * gf_g with GFLOPS per call, sorted, and returns how many rounds f won.
 */
static int race(mm_fn f, mm_fn g, const float* a, const float* b, float* c,
                int N, int reps, double* gf_f, double* gf_g) {
    f(a, b, c, N, N, N);
    g(a, b, c, N, N, N);
    int f_wins = 0;
    for (int r = 0; r < reps; r++) {
        double tf, tg;
        if (r % 2 == 0) {
            double t0 = now_ms(); f(a, b, c, N, N, N);
            double t1 = now_ms(); g(a, b, c, N, N, N);
            double t2 = now_ms();
            tf = t1 - t0; tg = t2 - t1;
        } else {
            double t0 = now_ms(); g(a, b, c, N, N, N);
            double t1 = now_ms(); f(a, b, c, N, N, N);
            double t2 = now_ms();
            tg = t1 - t0; tf = t2 - t1;
        }
        if (tf < tg) f_wins++;
        gf_f[r] = gflops(N, tf);
        gf_g[r] = gflops(N, tg);
    }
    qsort(gf_f, reps, sizeof(double), compare_doubles);
    qsort(gf_g, reps, sizeof(double), compare_doubles);
    return f_wins;
}
#endif


int main(int argc, char** argv) {
    /* The slivers assume 16 floats per streaming vector, as on the M4. */
    if ((int)svcntsw() * 2 != SLIVER) {
        printf("expected 16 floats per streaming vector, found %d\n", (int)svcntsw());
        return 1;
    }

    g_scratch = (float*)malloc(sizeof(float) * 2048 * 2048);
    g_apack   = (float*)malloc(sizeof(float) * MAX_MC * MAX_KC);
    g_bpack   = (float*)malloc(sizeof(float) * MAX_KC * MAX_NC);

    /* Memory is only really taken once a thread writes to its buffers. */
    for (int t = 0; t < MAX_THREADS; t++) {
        g_apack_thr[t] = (float*)malloc(sizeof(float) * MAX_MC * MAX_KC);
        g_bpack_thr[t] = (float*)malloc(sizeof(float) * MAX_KC * MAX_NC);
    }

    /* Shared packed A and B for the pool version. */
    g_ashared = (float*)malloc(sizeof(float) * SHARED_MAX * MAX_KC);
    g_bshared = (float*)malloc(sizeof(float) * SHARED_MAX * MAX_KC);

    /* One thread per performance core, counting this one. */
    int pool_size = 8;
#ifdef __APPLE__
    {
        int pcores = 0;
        size_t len = sizeof pcores;
        if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &len, NULL, 0) == 0 && pcores > 0) {
            pool_size = pcores;
        }
    }
#endif
    pool_start(pool_size);

    /* The last two shapes are big enough to cross the default block sizes. */
    enum { NUM_SHAPES = 11 };
    int shapes[NUM_SHAPES][3] = {
        {1, 1, 1}, {5, 7, 9}, {16, 16, 16}, {17, 19, 23}, {32, 32, 32},
        {33, 40, 31}, {48, 50, 47}, {100, 70, 90}, {130, 150, 140},
        {257, 513, 129}, {300, 600, 500},
    };
    int failures = 0;

    /* Packers must match pack_a byte for byte. */
    printf("\npacking, compared byte for byte with pack_a\n");
    printf("%-14s %10s %10s\n", "shape (m x k)", "blocked", "via ZA");
    printf("------------------------------------\n");
    for (int s = 0; s < NUM_SHAPES; s++) {
        int m = shapes[s][0], k = shapes[s][1];
        size_t count = (size_t)m * k;
        float* a = (float*)malloc(sizeof(float) * count);
        float* expected = (float*)malloc(sizeof(float) * count);
        float* got = (float*)malloc(sizeof(float) * count);
        fill(a, count, 3);
        pack_a(a, expected, m, k);

        memset(got, 0xff, sizeof(float) * count);
        pack_blocked(a, got, m, k);
        int blocked_ok = memcmp(got, expected, sizeof(float) * count) == 0;

        memset(got, 0xff, sizeof(float) * count);
        pack_za(a, got, m, k);
        int za_ok = memcmp(got, expected, sizeof(float) * count) == 0;

        if (!blocked_ok || !za_ok) failures++;
        printf("%4d x %-8d %10s %10s\n", m, k,
               blocked_ok ? "exact" : "WRONG", za_ok ? "exact" : "WRONG");
        free(a); free(expected); free(got);
    }
    printf("------------------------------------\n");

    /*
     * "tiny blocks" forces many chunks of k (KC = 5) and many small blocks,
     * so the path that reloads C into the tiles is tested on every shape.
     */
    printf("\ncorrectness, against a double-precision reference\n");
    printf("%-22s %9s %9s %9s %9s %9s %9s %9s %9s\n", "shape (m x k @ k x n)",
           "1 tile", "4 tiles", "blocked", "tiny blk", "3 thr", "3 thr tny",
           "pool 3", "pool7 tny");
    printf("------------------------------------------------------------------------------------------------------------\n");
    for (int s = 0; s < NUM_SHAPES; s++) {
        int m = shapes[s][0], k = shapes[s][1], n = shapes[s][2];
        size_t out_count = (size_t)m * n;
        float*  a   = (float*)malloc(sizeof(float) * (size_t)m * k);
        float*  b   = (float*)malloc(sizeof(float) * (size_t)k * n);
        float*  c   = (float*)malloc(sizeof(float) * out_count);
        double* ref = (double*)malloc(sizeof(double) * out_count);
        fill(a, (size_t)m * k, 1);
        fill(b, (size_t)k * n, 5);
        matmul_reference(a, b, ref, m, k, n);

        /* Poison before each kernel so unwritten entries show up. */
        pack_a(a, g_scratch, m, k);
        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        sme_kernel_1tile(g_scratch, b, c, m, k, n);
        double err1 = max_error(c, ref, out_count);

        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_sme(a, b, c, m, k, n);
        double err4 = max_error(c, ref, out_count);

        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_blocked(a, b, c, m, k, n);
        double err_blk = max_error(c, ref, out_count);

        g_kc = 5; g_mc = 64; g_nc = 64;
        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_blocked(a, b, c, m, k, n);
        double err_tiny = max_error(c, ref, out_count);
        g_kc = 256; g_mc = 256; g_nc = 2048;

        /* Three threads gives uneven strips, and none at all on tiny shapes. */
        g_threads = 3;
        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_threaded(a, b, c, m, k, n);
        double err_thr = max_error(c, ref, out_count);

        g_kc = 5; g_mc = 64; g_nc = 64;
        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_threaded(a, b, c, m, k, n);
        double err_thr_tiny = max_error(c, ref, out_count);
        g_kc = 256; g_mc = 256; g_nc = 2048;

        /* The pool: 3 threads, then 7 threads with tiny chunks and tiles,
         * which means many barriers and more threads than tiles on small shapes. */
        g_threads = 3;
        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_pool(a, b, c, m, k, n);
        double err_pool = max_error(c, ref, out_count);

        g_threads = 7;
        g_kc = 5; g_tile_m = 32; g_tile_n = 64;
        for (size_t i = 0; i < out_count; i++) c[i] = -999.0f;
        matmul_pool(a, b, c, m, k, n);
        double err_pool_tiny = max_error(c, ref, out_count);
        g_kc = 256; g_tile_m = 128; g_tile_n = 256;
        g_threads = 1;

        int ok = err1 < 1e-3 && err4 < 1e-3 && err_blk < 1e-3 && err_tiny < 1e-3
              && err_thr < 1e-3 && err_thr_tiny < 1e-3
              && err_pool < 1e-3 && err_pool_tiny < 1e-3;
        if (!ok) failures++;
        printf("%4d x %-4d @ %4d x %-4d %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e  %s\n",
               m, k, k, n, err1, err4, err_blk, err_tiny, err_thr, err_thr_tiny,
               err_pool, err_pool_tiny, ok ? "" : "FAIL");
        free(a); free(b); free(c); free(ref);
    }
    printf("------------------------------------------------------------------------------------------------------------\n");
    printf("failures: %d\n\n", failures);
    if (failures) { printf("not timing incorrect kernels\n"); return 1; }

#ifdef __APPLE__
    /* "./sme_matmul verify" skips the tables that are not needed to verify. */
    int verify_only = argc > 1 && strcmp(argv[1], "verify") == 0;

    /* Prefer performance cores (a request, not a guarantee). */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    /* Half a second of work so the CPU is at full clock before timing. */
    {
        size_t count = (size_t)1024 * 1024;
        float* a = (float*)malloc(sizeof(float) * count);
        float* b = (float*)malloc(sizeof(float) * count);
        float* c = (float*)malloc(sizeof(float) * count);
        fill(a, count, 1); fill(b, count, 5);
        double start = now_ms();
        while (now_ms() - start < 500.0) matmul_blocked(a, b, c, 1024, 1024, 1024);
        free(a); free(b); free(c);
    }

    /* Try every pair of block sizes at 2048 and keep the fastest. */
    {
        int kcs[4] = {256, 512, 1024, 2048};
        int mcs[4] = {64, 128, 256, 512};
        int N = 2048;
        size_t count = (size_t)N * N;
        float* a = (float*)malloc(sizeof(float) * count);
        float* b = (float*)malloc(sizeof(float) * count);
        float* c = (float*)malloc(sizeof(float) * count);
        fill(a, count, 1); fill(b, count, 5);

        double best = 1e30;
        int best_kc = g_kc, best_mc = g_mc;
        printf("tuning block sizes at %d x %d, GFLOPS (NC = %d)\n", N, N, g_nc);
        printf("%10s %8d %8d %8d %8d\n", "KC \\ MC", mcs[0], mcs[1], mcs[2], mcs[3]);
        printf("--------------------------------------------------\n");
        for (int i = 0; i < 4; i++) {
            printf("%10d", kcs[i]);
            for (int j = 0; j < 4; j++) {
                g_kc = kcs[i]; g_mc = mcs[j];
                double t = time_ms(matmul_blocked, a, b, c, N, N, N, 5);
                printf(" %8.0f", gflops(N, t));
                if (t < best) { best = t; best_kc = kcs[i]; best_mc = mcs[j]; }
            }
            printf("\n");
        }
        printf("--------------------------------------------------\n");
        g_kc = best_kc; g_mc = best_mc;
        printf("using KC = %d, MC = %d\n\n", g_kc, g_mc);
        free(a); free(b); free(c);
    }

    if (!verify_only) {
        int sizes[4] = {256, 512, 1024, 2048};
        int reps[4]  = {21, 11, 7, 5};
        double t_old[4], t_new[4], t_acc[4];
        for (int s = 0; s < 4; s++) {
            int N = sizes[s];
            size_t count = (size_t)N * N;
            float* a = (float*)malloc(sizeof(float) * count);
            float* b = (float*)malloc(sizeof(float) * count);
            float* c = (float*)malloc(sizeof(float) * count);
            fill(a, count, 1); fill(b, count, 5);
            t_old[s] = time_ms(matmul_sme,        a, b, c, N, N, N, reps[s]);
            t_new[s] = time_ms(matmul_blocked,    a, b, c, N, N, N, reps[s]);
            t_acc[s] = time_ms(matmul_accelerate, a, b, c, N, N, N, reps[s]);
            free(a); free(b); free(c);
        }

        printf("end to end against Accelerate\n");
        printf("%10s %10s %10s %10s %11s %11s\n", "size", "unblocked", "blocked",
               "blocked", "Accelerate", "Accelerate");
        printf("%10s %10s %10s %10s %11s %11s\n", "", "GFLOPS", "GFLOPS",
               "ms", "GFLOPS", "faster by");
        printf("------------------------------------------------------------------\n");
        for (int s = 0; s < 4; s++) {
            int N = sizes[s];
            printf("%4d x %-4d %10.0f %10.0f %10.2f %11.0f %10.1fx\n", N, N,
                   gflops(N, t_old[s]), gflops(N, t_new[s]), t_new[s],
                   gflops(N, t_acc[s]), t_new[s] / t_acc[s]);
        }
        printf("------------------------------------------------------------------\n");
        printf("Both SME kernels use one core. Accelerate appears to use several.\n\n");
    }

    /*
     * Tile size for the pool version, tried at 2048 with every pool thread.
     * Tiles must be whole slivers, so both sizes are multiples of 32.
     */
    {
        int tms[3] = {64, 128, 256};
        int tns[3] = {128, 256, 512};
        int N = 2048;
        size_t count = (size_t)N * N;
        float* a = (float*)malloc(sizeof(float) * count);
        float* b = (float*)malloc(sizeof(float) * count);
        float* c = (float*)malloc(sizeof(float) * count);
        fill(a, count, 1); fill(b, count, 5);

        g_threads = g_pool.size;
        double best = 1e30;
        int best_tm = g_tile_m, best_tn = g_tile_n;
        printf("tile size for the pool, %d x %d, %d threads, GFLOPS\n", N, N, g_threads);
        printf("%12s %8d %8d %8d\n", "rows \\ cols", tns[0], tns[1], tns[2]);
        printf("------------------------------------------\n");
        for (int i = 0; i < 3; i++) {
            printf("%12d", tms[i]);
            for (int j = 0; j < 3; j++) {
                g_tile_m = tms[i]; g_tile_n = tns[j];
                double t = time_ms(matmul_pool, a, b, c, N, N, N, 5);
                printf(" %8.0f", gflops(N, t));
                if (t < best) { best = t; best_tm = tms[i]; best_tn = tns[j]; }
            }
            printf("\n");
        }
        printf("------------------------------------------\n");
        g_tile_m = best_tm; g_tile_n = best_tn;
        g_threads = 1;
        printf("using tiles of %d x %d\n\n", g_tile_m, g_tile_n);
        free(a); free(b); free(c);
    }

    /*
     * Step 6 (new threads every call, equal strips) against Step 7 (pool,
     * shared packing, tiles handed out as threads become free).
     */
    if (!verify_only) {
        int tsizes[3] = {512, 1024, 2048};
        int treps[3]  = {11, 7, 5};
        int counts[8] = {1, 2, 3, 4, 5, 6, 8, 10};
        double gf_static[8][3], gf_pool[8][3], gf_acc[3];

        for (int s = 0; s < 3; s++) {
            int N = tsizes[s];
            size_t count = (size_t)N * N;
            float* a = (float*)malloc(sizeof(float) * count);
            float* b = (float*)malloc(sizeof(float) * count);
            float* c = (float*)malloc(sizeof(float) * count);
            fill(a, count, 1); fill(b, count, 5);
            for (int i = 0; i < 8; i++) {
                g_threads = counts[i];
                gf_static[i][s] = gflops(N, time_ms(matmul_threaded, a, b, c, N, N, N, treps[s]));
                gf_pool[i][s] = gflops(N, time_ms(matmul_pool, a, b, c, N, N, N, treps[s]));
            }
            gf_acc[s] = gflops(N, time_ms(matmul_accelerate, a, b, c, N, N, N, treps[s]));
            free(a); free(b); free(c);
        }
        g_threads = 1;

        printf("GFLOPS: step 6 (static strips) against step 7 (pool)\n");
        printf("%10s %17s %17s %17s\n", "threads", "512 x 512", "1024 x 1024", "2048 x 2048");
        printf("%10s", "");
        for (int s = 0; s < 3; s++) printf(" %8s %8s", "static", "pool");
        printf("\n");
        printf("----------------------------------------------------------------\n");
        for (int i = 0; i < 8; i++) {
            printf("%10d", counts[i]);
            for (int s = 0; s < 3; s++) printf(" %8.0f %8.0f", gf_static[i][s], gf_pool[i][s]);
            printf("\n");
        }
        printf("----------------------------------------------------------------\n");
        printf("%10s", "Accelerate");
        for (int s = 0; s < 3; s++) printf(" %17.0f", gf_acc[s]);
        printf("\n\n");
    }

    /*
     * Where the threads ran. Each line is one call at 2048. For every thread:
     * the CPU number when it started and when it finished, and how many
     * tiles it did. A thread that shares an SME unit should do fewer tiles.
     */
    if (!verify_only) {
        int N = 2048;
        size_t count = (size_t)N * N;
        float* a = (float*)malloc(sizeof(float) * count);
        float* b = (float*)malloc(sizeof(float) * count);
        float* c = (float*)malloc(sizeof(float) * count);
        fill(a, count, 1); fill(b, count, 5);

        int counts[3] = {2, 3, 4};
        printf("where the threads ran, %d x %d (cpu at start -> cpu at end, tiles done)\n", N, N);
        printf("%8s %4s %8s   %s\n", "threads", "run", "GFLOPS", "per thread");
        printf("--------------------------------------------------------------------------------\n");
        for (int i = 0; i < 3; i++) {
            g_threads = counts[i];
            matmul_pool(a, b, c, N, N, N); /* untimed warm-up */
            for (int run = 1; run <= 6; run++) {
                double start = now_ms();
                matmul_pool(a, b, c, N, N, N);
                double ms = now_ms() - start;
                printf("%8d %4d %8.0f  ", g_threads, run, gflops(N, ms));
                for (int t = 0; t < g_last_threads; t++) {
                    printf(" %2ld->%-2ld (%3d)", g_last_cpu_start[t], g_last_cpu_end[t],
                           g_last_tiles[t]);
                }
                printf("\n");
            }
        }
        printf("--------------------------------------------------------------------------------\n\n");
        g_threads = 1;
        free(a); free(b); free(c);
    }

    {
        enum { NV = 6, MAX_REPS = 101 };
        int vsizes[NV] = {512, 1000, 1024, 1536, 2000, 2048};
        int vreps[NV]  = {101, 41, 41, 41, 41, 41};
        double gf_ours[NV][MAX_REPS], gf_acc[NV][MAX_REPS];
        int wins[NV];

        g_threads = g_pool.size;
        printf("verification: %d threads, tiles %d x %d, KC = %d\n\n",
               g_threads, g_tile_m, g_tile_n, g_kc);

        printf("%-12s %-35s %s\n", "", "exact data", "random data in [-1, 1)");
        printf("%-12s %11s %11s %11s   %14s %11s %11s\n", "size",
               "same bits", "ours err", "Acc err", "max|ours-Acc|", "ours err", "Acc err");
        printf("-------------------------------------------------------------------------------------------\n");

        for (int s = 0; s < NV; s++) {
            int N = vsizes[s];
            size_t count = (size_t)N * N;
            float* a = (float*)malloc(sizeof(float) * count);
            float* b = (float*)malloc(sizeof(float) * count);
            float* c_ours = (float*)malloc(sizeof(float) * count);
            float* c_acc = (float*)malloc(sizeof(float) * count);

            /* 1. Exact data. */
            fill(a, count, 1); fill(b, count, 5);
            for (size_t i = 0; i < count; i++) { c_ours[i] = -999.0f; c_acc[i] = 999.0f; }
            matmul_pool(a, b, c_ours, N, N, N);
            matmul_accelerate(a, b, c_acc, N, N, N);
            size_t differ = 0;
            for (size_t i = 0; i < count; i++) {
                if (memcmp(&c_ours[i], &c_acc[i], sizeof(float)) != 0) differ++;
            }
            double ex_ours, ex_acc;
            spot_check(a, b, c_ours, c_acc, N, 4096, &ex_ours, &ex_acc);

            /* 2. Random data. */
            fill_random(a, count, 1); fill_random(b, count, 2);
            matmul_pool(a, b, c_ours, N, N, N);
            matmul_accelerate(a, b, c_acc, N, N, N);
            double gap = 0.0;
            for (size_t i = 0; i < count; i++) {
                double d = (double)c_ours[i] - c_acc[i];
                if (d < 0) d = -d;
                if (d > gap) gap = d;
            }
            double rnd_ours, rnd_acc;
            spot_check(a, b, c_ours, c_acc, N, 4096, &rnd_ours, &rnd_acc);

            char same[32];
            if (differ == 0) snprintf(same, sizeof same, "yes");
            else             snprintf(same, sizeof same, "NO (%zu)", differ);
            if (differ != 0 || ex_ours != 0.0 || ex_acc != 0.0) failures++;

            printf("%4d x %-6d %11s %11.1e %11.1e   %14.1e %11.1e %11.1e\n",
                   N, N, same, ex_ours, ex_acc, gap, rnd_ours, rnd_acc);

            /* 3. Timing, on the exact data as in every table before. */
            fill(a, count, 1); fill(b, count, 5);
            wins[s] = race(matmul_pool, matmul_accelerate, a, b, c_ours, N, vreps[s],
                           gf_ours[s], gf_acc[s]);

            free(a); free(b); free(c_ours); free(c_acc);
        }
        printf("-------------------------------------------------------------------------------------------\n");
        if (failures) printf("VERIFICATION FAILED: do not trust the timings below\n");
        printf("\n");

        printf("%-12s %-29s %-29s %8s %10s\n", "", "ours, GFLOPS", "Accelerate, GFLOPS",
               "medians", "ours won");
        printf("%-12s %9s %9s %9s  %9s %9s %9s %8s %10s\n", "size",
               "min", "median", "max", "min", "median", "max", "ours/Acc", "rounds");
        printf("-------------------------------------------------------------------------------------------\n");
        for (int s = 0; s < NV; s++) {
            int R = vreps[s];
            double med_ours = gf_ours[s][R / 2], med_acc = gf_acc[s][R / 2];
            printf("%4d x %-6d %9.0f %9.0f %9.0f  %9.0f %9.0f %9.0f %7.2fx %5d / %-3d\n",
                   vsizes[s], vsizes[s],
                   gf_ours[s][0], med_ours, gf_ours[s][R - 1],
                   gf_acc[s][0],  med_acc,  gf_acc[s][R - 1],
                   med_ours / med_acc, wins[s], R);
        }
        printf("-------------------------------------------------------------------------------------------\n\n");
        g_threads = 1;
    }
#else
    (void)argc; (void)argv;
    printf("(the speed comparison needs Accelerate, so it only runs on macOS)\n");
#endif

    free(g_scratch); free(g_apack); free(g_bpack);
    for (int t = 0; t < MAX_THREADS; t++) { free(g_apack_thr[t]); free(g_bpack_thr[t]); }
    pool_stop();
    free(g_ashared); free(g_bshared);
    return failures ? 1 : 0;
}