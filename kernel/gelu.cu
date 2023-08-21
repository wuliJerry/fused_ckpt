#include <cmath>

extern "C"
__global__ void gelu_unfused(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sigmoid_val = 1.0 / (1.0 + exp(-1.702 * x));
        output[idx] = x * sigmoid_val;
    }
}

extern "C"
__global__ void gelu_unfused_grad(float* input, float* grad_output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sigmoid_val = 1.0 / (1.0 + exp(-1.702 * x));
        float pdf = exp(-0.5 * x * x) * (1 / sqrt(2 * M_PI));
        grad_input[idx] = grad_output[idx] * (sigmoid_val + 1.702 * x * pdf);
    }
}

extern "C"
__global__ void gelu_fused(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float exp_val = exp(-1.702 * x);
        float sigmoid_val = 1.0 / (1.0 + exp_val);
        output[idx] = x * sigmoid_val;
    }
}

extern "C"
__global__ void gelu_fused_grad(float* input, float* grad_output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float exp_val = exp(-1.702 * x);
        float sigmoid_val = 1.0 / (1.0 + exp_val);
        float pdf = exp(-0.5 * x * x) * (1 / sqrt(2 * M_PI)); 
        grad_input[idx] = grad_output[idx] * (sigmoid_val + 1.702 * x * pdf);
    }
}

