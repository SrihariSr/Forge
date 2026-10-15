#include <arm_sme.h>
#include <stdio.h>


__arm_locally_streaming __arm_new("za")
static void outer_product(const float* x, const float* y, float* out, int* size_out)
{
    /*
     * How many floats fit in one streaming vector. Not fixed by the
     * instruction set: each chip chooses its own width. On an M4 it is 16
     * (512 bits), which makes one tile 16 by 16.
     */
    int n = (int)svcntsw();
    *size_out = n;

    /*
     * A predicate is a mask saying which lanes of a vector to use. This one
     * switches all lanes on. Masks matter later, for matrix edges that do not
     * fill a whole vector.
     */
    svbool_t all = svptrue_b32();

    /* Clear the whole matrix storage, so the tile starts at zero. */
    svzero_za();

    /* Load n floats of x and n floats of y into two streaming vectors. */
    svfloat32_t vx = svld1_f32(all, x);
    svfloat32_t vy = svld1_f32(all, y);

    svmopa_za32_f32_m(0, all, all, vx, vy);

    /*
     * Copy the tile back to ordinary memory one row at a time. "hor" means a
     * horizontal slice, i.e. a row. Row r of the tile lands at out[r * n].
     */
    for (int r = 0; r < n; r++) {
        svst1_hor_za32(0, r, all, out + r * n);
    }
}

int main(void)
{
    /* Room for up to 64 floats per vector, more than any chip uses today. */
    float x[64], y[64], out[64 * 64];

    /* Simple whole numbers, so every correct answer is exact. */
    for (int i = 0; i < 64; i++) {
        x[i] = (float)(i + 1);
        y[i] = (float)(10 * (i + 1));
    }

    int n = 0;
    outer_product(x, y, out, &n);

    /* Check every entry against the obvious answer, x[i] * y[j]. */
    int wrong = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (out[i * n + j] != x[i] * y[j]) wrong++;
        }
    }

    printf("streaming vector width     : %d floats\n", n);
    printf("tile size                  : %d x %d\n", n, n);
    printf("products in one instruction: %d\n", n * n);
    printf("wrong entries              : %d\n", wrong);
    printf("%s\n", wrong == 0 ? "SME works on this machine" : "SME gave wrong answers");

    return wrong ? 1 : 0;
}