#include "gpu_nms.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <chrono> // Added for timing measurements

#define MAX_DETECTIONS 4096
#define N_PARTITIONS 32

// CUDA kernel for generating NMS bitmap
__global__ void generate_nms_bitmap(float* boxes, float* confidences, uint8_t* nmsbitmap, 
                                   int num_boxes, float nms_threshold) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= num_boxes || j >= num_boxes) return;
    
    if (confidences[i] < confidences[j]) {
        // Calculate IoU
        float x1_i = boxes[i * 4];
        float y1_i = boxes[i * 4 + 1];
        float x2_i = boxes[i * 4] + boxes[i * 4 + 2];
        float y2_i = boxes[i * 4 + 1] + boxes[i * 4 + 3];
        
        float x1_j = boxes[j * 4];
        float y1_j = boxes[j * 4 + 1];
        float x2_j = boxes[j * 4] + boxes[j * 4 + 2];
        float y2_j = boxes[j * 4 + 1] + boxes[j * 4 + 3];
        
        float intersection_x1 = max(x1_i, x1_j);
        float intersection_y1 = max(y1_i, y1_j);
        float intersection_x2 = min(x2_i, x2_j);
        float intersection_y2 = min(y2_i, y2_j);
        
        float intersection_area = 0.0f;
        if (intersection_x2 > intersection_x1 && intersection_y2 > intersection_y1) {
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1);
        }
        
        float area_j = boxes[j * 4 + 2] * boxes[j * 4 + 3];
        float iou = (area_j > 0.0f) ? intersection_area / area_j : 0.0f;
        
        nmsbitmap[i * MAX_DETECTIONS + j] = (iou < nms_threshold) ? 1 : 0;
    } else {
        nmsbitmap[i * MAX_DETECTIONS + j] = 1;
    }
}

// CUDA kernel for reducing NMS bitmap
__global__ void reduce_nms_bitmap(uint8_t* nmsbitmap, uint8_t* pointsbitmap, int num_boxes) {
    int idx = blockIdx.x * MAX_DETECTIONS + threadIdx.x;
    
    if (threadIdx.x < num_boxes) {
        uint8_t result = nmsbitmap[idx];
        
        for (int i = 1; i < N_PARTITIONS && (idx + i * MAX_DETECTIONS / N_PARTITIONS) < MAX_DETECTIONS; i++) {
            result = result && nmsbitmap[idx + i * MAX_DETECTIONS / N_PARTITIONS];
        }
        
        pointsbitmap[blockIdx.x] = result;
    }
}

GPUNMS::GPUNMS() : initialized_(false), max_detections_(MAX_DETECTIONS),
                   gpu_boxes_(nullptr), gpu_confidences_(nullptr),
                   gpu_nms_bitmap_(nullptr), gpu_points_bitmap_(nullptr),
                   cpu_points_bitmap_(nullptr) {
}

GPUNMS::~GPUNMS() {
    cleanup();
}

bool GPUNMS::init(int max_detections) {
    if (initialized_) {
        cleanup();
    }
    
    max_detections_ = min(max_detections, MAX_DETECTIONS);
    
    // Create CUDA events
    cudaEventCreate(&begin_event_);
    cudaEventCreate(&end_event_);
    
    // Allocate GPU memory
    cudaError_t err;
    
    err = cudaMalloc((void**)&gpu_boxes_, sizeof(float) * max_detections_ * 4);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to allocate GPU boxes memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc((void**)&gpu_confidences_, sizeof(float) * max_detections_);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to allocate GPU confidences memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc((void**)&gpu_nms_bitmap_, sizeof(uint8_t) * MAX_DETECTIONS * MAX_DETECTIONS);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to allocate GPU NMS bitmap memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc((void**)&gpu_points_bitmap_, sizeof(uint8_t) * MAX_DETECTIONS);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to allocate GPU points bitmap memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Allocate CPU memory
    cpu_points_bitmap_ = (uint8_t*)malloc(sizeof(uint8_t) * MAX_DETECTIONS);
    if (!cpu_points_bitmap_) {
        printf("GPU NMS Error: Failed to allocate CPU points bitmap memory\n");
        return false;
    }
    
    initialized_ = true;
    return true;
}

std::vector<int> GPUNMS::runNMS(const std::vector<cv::Rect>& boxes, 
                                const std::vector<float>& confidences, 
                                float nms_threshold) {
    if (!initialized_ || boxes.empty()) {
        printf("GPU NMS: Not initialized or empty boxes\n");
        return std::vector<int>();
    }
    
    int num_boxes = min((int)boxes.size(), max_detections_);
    printf("GPU NMS: Processing %d boxes\n", num_boxes);
    
    // Start total timing
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Prepare data for GPU
    std::vector<float> boxes_data;
    std::vector<float> confidences_data;
    std::vector<int> indices;
    
    boxes_data.reserve(num_boxes * 4);
    confidences_data.reserve(num_boxes);
    indices.reserve(num_boxes);
    
    // Create index array and sort by confidence
    for (int i = 0; i < num_boxes; i++) {
        indices.push_back(i);
    }
    
    std::sort(indices.begin(), indices.end(), [&confidences](int a, int b) {
        return confidences[a] > confidences[b];
    });
    
    // Prepare sorted data for GPU
    for (int i = 0; i < num_boxes; i++) {
        int idx = indices[i];
        const cv::Rect& box = boxes[idx];
        
        boxes_data.push_back((float)box.x);
        boxes_data.push_back((float)box.y);
        boxes_data.push_back((float)box.width);
        boxes_data.push_back((float)box.height);
        
        confidences_data.push_back(confidences[idx]);
    }
    
    auto prep_end = std::chrono::high_resolution_clock::now();
    double prep_time = std::chrono::duration<double, std::milli>(prep_end - total_start).count();
    printf("GPU NMS: Data preparation: %.3f ms\n", prep_time);
    
    // Copy data to GPU
    cudaError_t err;
    err = cudaMemcpy(gpu_boxes_, boxes_data.data(), sizeof(float) * num_boxes * 4, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to copy boxes to GPU: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    err = cudaMemcpy(gpu_confidences_, confidences_data.data(), sizeof(float) * num_boxes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to copy confidences to GPU: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    auto copy_end = std::chrono::high_resolution_clock::now();
    double copy_time = std::chrono::duration<double, std::milli>(copy_end - prep_end).count();
    printf("GPU NMS: GPU memory copy: %.3f ms\n", copy_time);
    
    // Initialize NMS bitmap
    err = cudaMemset(gpu_nms_bitmap_, 1, sizeof(uint8_t) * MAX_DETECTIONS * MAX_DETECTIONS);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to initialize NMS bitmap: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    // Initialize points bitmap
    err = cudaMemset(gpu_points_bitmap_, 0, sizeof(uint8_t) * MAX_DETECTIONS);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to initialize points bitmap: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    // Launch NMS kernels
    dim3 block_dim(16, 16);
    dim3 grid_dim((num_boxes + block_dim.x - 1) / block_dim.x, 
                  (num_boxes + block_dim.y - 1) / block_dim.y);
    
    cudaEventRecord(begin_event_);
    
    generate_nms_bitmap<<<grid_dim, block_dim>>>(gpu_boxes_, gpu_confidences_, 
                                                gpu_nms_bitmap_, num_boxes, nms_threshold);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to launch generate_nms_bitmap kernel: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    // Launch reduce kernel
    dim3 reduce_block_dim(MAX_DETECTIONS / N_PARTITIONS);
    dim3 reduce_grid_dim(num_boxes);
    
    reduce_nms_bitmap<<<reduce_grid_dim, reduce_block_dim>>>(gpu_nms_bitmap_, 
                                                            gpu_points_bitmap_, num_boxes);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to launch reduce_nms_bitmap kernel: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    cudaEventRecord(end_event_);
    cudaEventSynchronize(end_event_);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, begin_event_, end_event_);
    printf("GPU NMS: Kernel execution: %.3f ms\n", kernel_time);
    
    // Copy results back to CPU
    err = cudaMemcpy(cpu_points_bitmap_, gpu_points_bitmap_, 
                    sizeof(uint8_t) * MAX_DETECTIONS, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("GPU NMS Error: Failed to copy results from GPU: %s\n", cudaGetErrorString(err));
        return std::vector<int>();
    }
    
    auto copy_back_end = std::chrono::high_resolution_clock::now();
    double copy_back_time = std::chrono::duration<double, std::milli>(copy_back_end - copy_end).count();
    printf("GPU NMS: GPU to CPU copy: %.3f ms\n", copy_back_time);
    
    // Build result vector
    std::vector<int> result;
    for (int i = 0; i < num_boxes; i++) {
        if (cpu_points_bitmap_[i]) {
            result.push_back(indices[i]); // Return original indices
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    printf("GPU NMS: Total time: %.3f ms, Results: %d\n", total_time, (int)result.size());
    
    return result;
}

void GPUNMS::cleanup() {
    if (initialized_) {
        cudaEventDestroy(begin_event_);
        cudaEventDestroy(end_event_);
        
        if (gpu_boxes_) cudaFree(gpu_boxes_);
        if (gpu_confidences_) cudaFree(gpu_confidences_);
        if (gpu_nms_bitmap_) cudaFree(gpu_nms_bitmap_);
        if (gpu_points_bitmap_) cudaFree(gpu_points_bitmap_);
        if (cpu_points_bitmap_) free(cpu_points_bitmap_);
        
        gpu_boxes_ = nullptr;
        gpu_confidences_ = nullptr;
        gpu_nms_bitmap_ = nullptr;
        gpu_points_bitmap_ = nullptr;
        cpu_points_bitmap_ = nullptr;
        
        initialized_ = false;
    }
} 