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

void matmul(const float* a, const float* b, float* out, int m, int n, int p){
    for (int i = 0; i < m; i++){ /* Each output row */
        for (int j = 0; j < n; j++){ /* Each output column */
            float sum = 0.0f;
            for (int k = 0; k < p; k++){ /* Dot product over shared dim */
                sum += a[i*p + k] * b[k*p + j];
            }
            out[i*n + j] = sum;
        }
    }
}
