#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>
#include <cmath>


static Logger logger;
#define isFP16 true
#define warmup true


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, logger);
    }
    // Build an engine from an onnx model
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}


void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
#if NV_TENSORRT_MAJOR < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
#endif
    num_classes = detection_attribute_size - 4;

    // Initialize input buffers
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize GPU NMS
    if (!gpu_nms_.init(num_detections)) {
        printf("Warning: GPU NMS initialization failed\n");
    } else {
        printf("GPU NMS initialized successfully\n");
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

YOLOv11::~YOLOv11()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image) {
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer()
{
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    // Set input and output tensor addresses for TensorRT 10+
    context->setTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
    this->context->enqueueV3(this->stream);
#endif
}

void YOLOv11::postprocess(vector<Detection>& output)
{
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Debug: Print the size of data being copied
    size_t data_size = num_detections * detection_attribute_size * sizeof(float);
    std::cout << "Copying " << data_size << " bytes (" << (data_size / 1024.0 / 1024.0) << " MB) from GPU to CPU" << std::endl;
    std::cout << "num_detections: " << num_detections << ", detection_attribute_size: " << detection_attribute_size << std::endl;
    
    // Memcpy from device output buffer to host output buffer
    auto start_memcpy = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(end_memcpy - start_memcpy).count();

    auto start_parse = std::chrono::high_resolution_clock::now();
    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
            
            // Debug: Print detection info
            printf("Detection %d: class=%s, conf=%.3f, box=[%d,%d,%d,%d]\n", 
                   (int)boxes.size()-1, CLASS_NAMES[class_id_point.y].c_str(), score, box.x, box.y, box.width, box.height);
        }
    }
    auto end_parse = std::chrono::high_resolution_clock::now();
    auto parse_time = std::chrono::duration_cast<std::chrono::microseconds>(end_parse - start_parse).count();

    auto start_nms = std::chrono::high_resolution_clock::now();
    vector<int> nms_result = gpu_nms_.runNMS(boxes, confidences, nms_threshold);
    auto end_nms = std::chrono::high_resolution_clock::now();
    auto nms_time = std::chrono::duration_cast<std::chrono::microseconds>(end_nms - start_nms).count();

    auto start_finalize = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
        
        // Debug: Print final detection info
        printf("Final Detection %d: class=%s, conf=%.3f, box=[%d,%d,%d,%d]\n", 
               i, CLASS_NAMES[result.class_id].c_str(), result.conf, result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height);
    }
    auto end_finalize = std::chrono::high_resolution_clock::now();
    auto finalize_time = std::chrono::duration_cast<std::chrono::microseconds>(end_finalize - start_finalize).count();
    
    // COMMENTED OUT: Apply clustering to merge nearby detections
    // auto start_cluster = std::chrono::high_resolution_clock::now();
    // output = clusterDetections(output, 50.0f);  // Cluster detections within 50 pixels
    // auto end_cluster = std::chrono::high_resolution_clock::now();
    // auto cluster_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cluster - start_cluster).count();
    
    // COMMENTED OUT: Print final clustered detections
    // printf("=== CLUSTERED RESULTS ===\n");
    // for (int i = 0; i < output.size(); i++) {
    //     printf("Cluster %d: class=%s, conf=%.3f, box=[%d,%d,%d,%d]\n", 
    //            i, CLASS_NAMES[output[i].class_id].c_str(), output[i].conf, 
    //            output[i].bbox.x, output[i].bbox.y, output[i].bbox.width, output[i].bbox.height);
    // }
    
    // Print final NMS results (without clustering)
    printf("=== FINAL NMS RESULTS ===\n");
    for (int i = 0; i < output.size(); i++) {
        printf("Result %d: class=%s, conf=%.3f, box=[%d,%d,%d,%d]\n", 
               i, CLASS_NAMES[output[i].class_id].c_str(), output[i].conf, 
               output[i].bbox.x, output[i].bbox.y, output[i].bbox.width, output[i].bbox.height);
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
    
    std::cout << "Postprocess timing breakdown:" << std::endl;
    std::cout << "  Memcpy: " << memcpy_time << " μs (" << (memcpy_time * 100.0 / total_time) << "%)" << std::endl;
    std::cout << "  Parse: " << parse_time << " μs (" << (parse_time * 100.0 / total_time) << "%)" << std::endl;
    std::cout << "  NMS: " << nms_time << " μs (" << (nms_time * 100.0 / total_time) << "%)" << std::endl;
    std::cout << "  Finalize: " << finalize_time << " μs (" << (finalize_time * 100.0 / total_time) << "%)" << std::endl;
    // std::cout << "  Clustering: " << cluster_time << " μs (" << (cluster_time * 100.0 / total_time) << "%)" << std::endl;
    std::cout << "  Total: " << total_time << " μs" << std::endl;
}

vector<int> YOLOv11::fastNMS(const vector<Rect>& boxes, const vector<float>& confidences, float nms_threshold)
{
    if (boxes.empty()) return {};
    
    vector<int> indices(boxes.size());
    for (int i = 0; i < boxes.size(); i++) {
        indices[i] = i;
    }
    
    // Sort by confidence (descending)
    sort(indices.begin(), indices.end(), [&confidences](int a, int b) {
        return confidences[a] > confidences[b];
    });
    
    vector<int> keep;
    keep.reserve(boxes.size());
    
    // Use a more efficient NMS algorithm
    for (int i = 0; i < indices.size(); i++) {
        int current_idx = indices[i];
        bool should_keep = true;
        
        // Check IoU with already kept boxes
        for (int kept_idx : keep) {
            float iou = calculateIoU(boxes[current_idx], boxes[kept_idx]);
            if (iou > nms_threshold) {
                should_keep = false;
                break;
            }
        }
        
        if (should_keep) {
            keep.push_back(current_idx);
        }
    }
    
    return keep;
}

float YOLOv11::calculateIoU(const Rect& box1, const Rect& box2)
{
    // Calculate intersection
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    int intersection = (x2 - x1) * (y2 - y1);
    
    // Calculate union
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;
    int union_area = area1 + area2 - intersection;
    
    return static_cast<float>(intersection) / static_cast<float>(union_area);
}

vector<Detection> YOLOv11::clusterDetections(const vector<Detection>& detections, float cluster_threshold) {
    if (detections.empty()) return detections;
    
    vector<Detection> clustered_detections;
    vector<bool> used(detections.size(), false);
    
    for (int i = 0; i < detections.size(); i++) {
        if (used[i]) continue;
        
        Detection best_detection = detections[i];
        used[i] = true;
        
        // Find all detections near this one
        for (int j = i + 1; j < detections.size(); j++) {
            if (used[j]) continue;
            
            // Calculate center distance
            float center1_x = detections[i].bbox.x + detections[i].bbox.width / 2.0f;
            float center1_y = detections[i].bbox.y + detections[i].bbox.height / 2.0f;
            float center2_x = detections[j].bbox.x + detections[j].bbox.width / 2.0f;
            float center2_y = detections[j].bbox.y + detections[j].bbox.height / 2.0f;
            
            float distance = sqrt((center1_x - center2_x) * (center1_x - center2_x) + 
                                (center1_y - center2_y) * (center1_y - center2_y));
            
            if (distance < cluster_threshold) {
                // Keep the detection with higher confidence
                if (detections[j].conf > best_detection.conf) {
                    best_detection = detections[j];
                }
                used[j] = true;
            }
        }
        
        clustered_detections.push_back(best_detection);
    }
    
    printf("Clustering: Reduced %d detections to %d clusters\n", (int)detections.size(), (int)clustered_detections.size());
    return clustered_detections;
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
#if NV_TENSORRT_MAJOR < 10
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#else
    INetworkDefinition* network = builder->createNetworkV2(0U);
#endif
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }

    // Resize image to 640x480 for display
    cv::resize(image, image, cv::Size(640, 480));
}