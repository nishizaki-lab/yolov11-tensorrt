#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include "gpu_nms.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class YOLOv11
{

public:

    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;

    cudaStream_t stream;
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.5f;  // Increased from 0.3f - more selective
    float nms_threshold = 0.1f;   // Decreased from 0.3f - very aggressive suppression

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
    
    // Optimized NMS functions
    vector<int> fastNMS(const vector<Rect>& boxes, const vector<float>& confidences, float nms_threshold);
    float calculateIoU(const Rect& box1, const Rect& box2);
    
    // Clustering function to merge nearby detections
    vector<Detection> clusterDetections(const vector<Detection>& detections, float cluster_threshold = 50.0f);
    
    // GPU NMS
    GPUNMS gpu_nms_;
};