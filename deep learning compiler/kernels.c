void fused(const float* a, const float* b, const float* c, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float t = a[i] + b[i];
        t = t > 0.0f ? t : 0.0f;
        out[i] = t * c[i];
    }
}