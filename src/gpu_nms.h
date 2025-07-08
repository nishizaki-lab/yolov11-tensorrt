#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

// GPU NMS interface
class GPUNMS {
public:
    GPUNMS();
    ~GPUNMS();
    
    // Initialize GPU memory
    bool init(int max_detections = 4096);
    
    // Run GPU NMS on detection boxes
    std::vector<int> runNMS(const std::vector<cv::Rect>& boxes, 
                           const std::vector<float>& confidences, 
                           float nms_threshold = 0.3f);
    
    // Clean up GPU memory
    void cleanup();

private:
    bool initialized_;
    int max_detections_;
    
    // GPU memory pointers
    float* gpu_boxes_;
    float* gpu_confidences_;
    uint8_t* gpu_nms_bitmap_;
    uint8_t* gpu_points_bitmap_;
    
    // CPU memory for results
    uint8_t* cpu_points_bitmap_;
    
    // CUDA events for timing
    cudaEvent_t begin_event_;
    cudaEvent_t end_event_;
}; 