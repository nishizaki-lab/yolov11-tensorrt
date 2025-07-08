#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "YOLOv11.h"


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_file> <input_source>" << std::endl;
        std::cout << "  model_file: .onnx or .engine file" << std::endl;
        std::cout << "  input_source: image file, video file, folder path, or camera index (0,1,2...)" << std::endl;
        return -1;
    }

    const string model_file_path{ argv[1] };
    const string path{ argv[2] };
    
    // Check if model file exists
    if (!IsFile(model_file_path)) {
        std::cout << "Error: Model file '" << model_file_path << "' does not exist!" << std::endl;
        return -1;
    }
    
    // Check if it's an ONNX file and inform user about engine conversion
    bool is_onnx = model_file_path.find(".onnx") != std::string::npos;
    if (is_onnx) {
        std::cout << "ONNX file detected. Converting to TensorRT engine..." << std::endl;
        
        // Create expected engine file path
        string engine_path = model_file_path;
        size_t dotIndex = engine_path.find_last_of(".");
        if (dotIndex != std::string::npos) {
            engine_path = engine_path.substr(0, dotIndex) + ".engine";
        }
        std::cout << "Engine file will be saved as: " << engine_path << std::endl;
    }

    vector<string> imagePathList;
    bool                     isVideo{ false };
    bool                     isCamera{ false };
    int                      cameraIndex{ 0 };

    // Check if the second argument is a camera index (numeric)
    if (path.find_first_not_of("0123456789") == string::npos) {
        isCamera = true;
        cameraIndex = stoi(path);
    }
    else if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    // Initialize model - this will automatically handle ONNX to engine conversion
    std::cout << "Initializing model..." << std::endl;
    YOLOv11 model(model_file_path, logger);
    std::cout << "Model initialized successfully!" << std::endl;
    
    if (is_onnx) {
        std::cout << "TensorRT engine has been created and saved." << std::endl;
        std::cout << "Next time you can use the .engine file directly for faster loading." << std::endl;
    }

    // Timing variables for FPS calculation
    auto frame_start = std::chrono::high_resolution_clock::now();
    auto frame_end = std::chrono::high_resolution_clock::now();
    double total_time = 0.0;
    int frame_count = 0;
    double fps = 0.0;

    if (isCamera) {
        // Open camera device
        cv::VideoCapture cap(cameraIndex);
        if (!cap.isOpened()) {
            printf("Error: Could not open camera %d\n", cameraIndex);
            return -1;
        }
        
        printf("Camera %d opened successfully\n", cameraIndex);

        while (1)
        {
            frame_start = std::chrono::high_resolution_clock::now();
            
            Mat image;
            cap >> image;

            if (image.empty()) {
                printf("Error: Could not read frame from camera\n");
                break;
            }

            vector<Detection> objects;
            
            // Timing for preprocessing
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            model.preprocess(image);
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

            // Timing for inference
            auto infer_start = std::chrono::high_resolution_clock::now();
            model.infer();
            auto infer_end = std::chrono::high_resolution_clock::now();
            double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

            // Timing for postprocessing
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            model.postprocess(objects);
            auto postprocess_end = std::chrono::high_resolution_clock::now();
            double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

            // Draw detections and timing info
            model.draw(image, objects);
            
            // Calculate total time and FPS
            frame_end = std::chrono::high_resolution_clock::now();
            double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            total_time += frame_time;
            frame_count++;
            
            if (frame_count % 30 == 0) { // Update FPS every 30 frames
                fps = 1000.0 / (total_time / frame_count);
            }

            // Display timing information on image
            string timing_text = "FPS: " + to_string(static_cast<int>(fps));
            string preprocess_text = "Preprocess: " + to_string(preprocess_time).substr(0, 6) + "ms";
            string infer_text = "Inference: " + to_string(infer_time).substr(0, 6) + "ms";
            string postprocess_text = "Postprocess: " + to_string(postprocess_time).substr(0, 6) + "ms";
            string total_text = "Total: " + to_string(frame_time).substr(0, 6) + "ms";
            
            // Draw timing info on image
            cv::putText(image, timing_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(image, preprocess_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            cv::putText(image, infer_text, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            cv::putText(image, postprocess_text, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            cv::putText(image, total_text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            // Print timing to console
            printf("Frame %d - FPS: %.1f | Preprocess: %.2fms | Inference: %.2fms | Postprocess: %.2fms | Total: %.2fms\n", 
                   frame_count, fps, preprocess_time, infer_time, postprocess_time, frame_time);

            imshow("prediction", image);
            char key = waitKey(1);
            if (key == 27) { // ESC key to exit
                break;
            }
        }

        // Release resources
        destroyAllWindows();
        cap.release();
    }
    else if (isVideo) {
        //path to video
        cv::VideoCapture cap(path);

        while (1)
        {
            frame_start = std::chrono::high_resolution_clock::now();
            
            Mat image;
            cap >> image;

            if (image.empty()) break;

            vector<Detection> objects;
            
            // Timing for preprocessing
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            model.preprocess(image);
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

            // Timing for inference
            auto infer_start = std::chrono::high_resolution_clock::now();
            model.infer();
            auto infer_end = std::chrono::high_resolution_clock::now();
            double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

            // Timing for postprocessing
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            model.postprocess(objects);
            auto postprocess_end = std::chrono::high_resolution_clock::now();
            double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

            // Draw detections and timing info
            model.draw(image, objects);
            
            // Calculate total time and FPS
            frame_end = std::chrono::high_resolution_clock::now();
            double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            total_time += frame_time;
            frame_count++;
            
            if (frame_count % 30 == 0) { // Update FPS every 30 frames
                fps = 1000.0 / (total_time / frame_count);
            }

            // Display timing information on image
            string timing_text = "FPS: " + to_string(static_cast<int>(fps));
            string preprocess_text = "Preprocess: " + to_string(preprocess_time).substr(0, 6) + "ms";
            string infer_text = "Inference: " + to_string(infer_time).substr(0, 6) + "ms";
            string postprocess_text = "Postprocess: " + to_string(postprocess_time).substr(0, 6) + "ms";
            string total_text = "Total: " + to_string(frame_time).substr(0, 6) + "ms";
            
            // Draw timing info on image
            cv::putText(image, timing_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(image, preprocess_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            cv::putText(image, infer_text, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            cv::putText(image, postprocess_text, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            cv::putText(image, total_text, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            imshow("prediction", image);
            waitKey(1);
        }

        // Release resources
        destroyAllWindows();
        cap.release();
    }
    else {
        // path to folder saves images
        for (const auto& imagePath : imagePathList)
        {
            frame_start = std::chrono::high_resolution_clock::now();
            
            // open image
            Mat image = imread(imagePath);
            if (image.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            vector<Detection> objects;
            
            // Timing for preprocessing
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            model.preprocess(image);
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

            // Timing for inference
            auto infer_start = std::chrono::high_resolution_clock::now();
            model.infer();
            auto infer_end = std::chrono::high_resolution_clock::now();
            double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

            // Timing for postprocessing
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            model.postprocess(objects);
            auto postprocess_end = std::chrono::high_resolution_clock::now();
            double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

            // Draw detections and timing info
            model.draw(image, objects);
            
            // Calculate total time
            frame_end = std::chrono::high_resolution_clock::now();
            double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

            // Display timing information on image
            string preprocess_text = "Preprocess: " + to_string(preprocess_time).substr(0, 6) + "ms";
            string infer_text = "Inference: " + to_string(infer_time).substr(0, 6) + "ms";
            string postprocess_text = "Postprocess: " + to_string(postprocess_time).substr(0, 6) + "ms";
            string total_text = "Total: " + to_string(frame_time).substr(0, 6) + "ms";
            
            // Draw timing info on image
            cv::putText(image, preprocess_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            cv::putText(image, infer_text, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            cv::putText(image, postprocess_text, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            cv::putText(image, total_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            // Print timing to console
            printf("Image: %s | Preprocess: %.2fms | Inference: %.2fms | Postprocess: %.2fms | Total: %.2fms\n", 
                   imagePath.c_str(), preprocess_time, infer_time, postprocess_time, frame_time);

            imshow("Result", image);
            waitKey(0);
        }
    }

    return 0;
}