// Comparison_and_Analysis.cpp

#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp> // For Universal Intrinsics
#include <iostream>
#include <thread>
#include <vector>
#include <immintrin.h> // For AVX
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// 模板函数以支持不同的数据类型
template <typename T>
inline T clamp_value(double value) {
    return static_cast<T>(std::min(std::max(value, static_cast<double>(std::numeric_limits<T>::min())),
                                  static_cast<double>(std::numeric_limits<T>::max())));
}

// 最近邻插值模板函数
template <typename T>
inline T nearestNeighbor(const Mat& input, double x, double y) {
    int x_src = static_cast<int>(round(x));
    int y_src = static_cast<int>(round(y));
    x_src = std::min(std::max(x_src, 0), input.cols - 1);
    y_src = std::min(std::max(y_src, 0), input.rows - 1);
    return input.at<T>(y_src, x_src);
}

// 双线性插值模板函数
template <typename T>
inline T bilinearInterpolate(const Mat& input, double x, double y) {
    int x1 = static_cast<int>(floor(x));
    int y1 = static_cast<int>(floor(y));
    int x2 = std::min(x1 + 1, input.cols - 1);
    int y2 = std::min(y1 + 1, input.rows - 1);

    double dx = x - x1;
    double dy = y - y1;

    double top = (1.0 - dx) * input.at<T>(y1, x1) + dx * input.at<T>(y1, x2);
    double bottom = (1.0 - dx) * input.at<T>(y2, x1) + dx * input.at<T>(y2, x2);
    double value = (1.0 - dy) * top + dy * bottom;

    return clamp_value<T>(value);
}

// 双线性插值（多通道）模板函数
template <typename T>
inline Vec<T, 3> bilinearInterpolateVec(const Mat& input, double x, double y) {
    int x1 = static_cast<int>(floor(x));
    int y1 = static_cast<int>(floor(y));
    int x2 = std::min(x1 + 1, input.cols - 1);
    int y2 = std::min(y1 + 1, input.rows - 1);

    double dx = x - x1;
    double dy = y - y1;

    Vec<double, 3> top, bottom, value;
    for(int c = 0; c < 3; ++c){
        top[c] = (1.0 - dx) * input.at<Vec<T,3>>(y1, x1)[c] + dx * input.at<Vec<T,3>>(y1, x2)[c];
        bottom[c] = (1.0 - dx) * input.at<Vec<T,3>>(y2, x1)[c] + dx * input.at<Vec<T,3>>(y2, x2)[c];
        value[c] = (1.0 - dy) * top[c] + dy * bottom[c];
        // Clamp value
        value[c] = std::min(std::max(value[c], static_cast<double>(std::numeric_limits<T>::min())),
                            static_cast<double>(std::numeric_limits<T>::max()));
    }

    return Vec<T, 3>(static_cast<T>(value[0]), static_cast<T>(value[1]), static_cast<T>(value[2]));
}

// SIMD优化的双线性插值（仅适用于CV_32FC3）
Mat bilinearResizeAVX(const Mat& input_image, int output_width, int output_height) {
    CV_Assert(input_image.type() == CV_32FC3); // 仅支持CV_32FC3

    Mat output_image(output_height, output_width, input_image.type());

    double scale_width = static_cast<double>(input_image.cols) / output_width;
    double scale_height = static_cast<double>(input_image.rows) / output_height;

    parallel_for_(Range(0, output_height), [&](const Range& range) {
        for(int y_dst = range.start; y_dst < range.end; ++y_dst) {
            for(int x_dst = 0; x_dst < output_width; x_dst += 8) { // 处理8个像素
                // 计算源坐标
                double x_src_f_base = x_dst * scale_width;
                double y_src_f = y_dst * scale_height;

                int x1_base = static_cast<int>(floor(x_src_f_base));
                int y1 = static_cast<int>(floor(y_src_f));
                int x2_base = std::min(x1_base + 1, input_image.cols - 1);
                int y2 = std::min(y1 + 1, input_image.rows - 1);

                float dy = static_cast<float>(y_src_f - y1);
                __m256 _dy = _mm256_set1_ps(dy);
                __m256 one_minus_dy = _mm256_set1_ps(1.0f - dy);

                for(int i = 0; i < 8; ++i) {
                    double x_src_f = x_src_f_base + i * scale_width;
                    int x1 = static_cast<int>(floor(x_src_f));
                    int x2 = std::min(x1 + 1, input_image.cols - 1);
                    float dx = static_cast<float>(x_src_f - x1);

                    __m256 _dx = _mm256_set1_ps(dx);
                    __m256 one_minus_dx = _mm256_set1_ps(1.0f - dx);

                    // 使用AVX指令计算每个通道的值
                    for(int c = 0; c < 3; ++c) {
                        // 读取像素值
                        float p1 = input_image.at<Vec3f>(y1, x1)[c];
                        float p2 = input_image.at<Vec3f>(y1, x2)[c];
                        float p3 = input_image.at<Vec3f>(y2, x1)[c];
                        float p4 = input_image.at<Vec3f>(y2, x2)[c];

                        // 加载到AVX寄存器
                        __m256 val_p1 = _mm256_set1_ps(p1);
                        __m256 val_p2 = _mm256_set1_ps(p2);
                        __m256 val_p3 = _mm256_set1_ps(p3);
                        __m256 val_p4 = _mm256_set1_ps(p4);

                        // top = (1 - dx) * p1 + dx * p2
                        __m256 top = _mm256_add_ps(_mm256_mul_ps(one_minus_dx, val_p1), _mm256_mul_ps(_dx, val_p2));

                        // bottom = (1 - dx) * p3 + dx * p4
                        __m256 bottom = _mm256_add_ps(_mm256_mul_ps(one_minus_dx, val_p3), _mm256_mul_ps(_dx, val_p4));

                        // value = (1 - dy) * top + dy * bottom
                        __m256 value = _mm256_add_ps(_mm256_mul_ps(one_minus_dy, top), _mm256_mul_ps(_dy, bottom));

                        // 存储结果
                        _mm256_storeu_ps(&output_image.at<Vec3f>(y_dst, x_dst + i)[c], value);
                    }
                }
            }
        }
    });

    return output_image;
}

// 基于OpenCV Universal Intrinsics的双线性插值
Mat bilinearResizeUniversal(const Mat& input_image, int output_width, int output_height) {
    // 目前仅实现CV_32FC3的版本
    CV_Assert(input_image.type() == CV_32FC3); // 仅支持CV_32FC3

    Mat output_image(output_height, output_width, input_image.type());

    double scale_width = static_cast<double>(input_image.cols) / output_width;
    double scale_height = static_cast<double>(input_image.rows) / output_height;

    parallel_for_(Range(0, output_height), [&](const Range& range) {
        for(int y_dst = range.start; y_dst < range.end; ++y_dst) {
            for(int x_dst = 0; x_dst < output_width; x_dst += 8) { // 处理8个像素
                // 计算源坐标
                double x_src_f_base = x_dst * scale_width;
                double y_src_f = y_dst * scale_height;

                int x1_base = static_cast<int>(floor(x_src_f_base));
                int y1 = static_cast<int>(floor(y_src_f));
                int x2_base = std::min(x1_base + 1, input_image.cols - 1);
                int y2 = std::min(y1 + 1, input_image.rows - 1);

                float dy = static_cast<float>(y_src_f - y1);
                float one_minus_dy_f = 1.0f - dy;

                // 使用OpenCV Universal Intrinsics加载和计算
                for(int i = 0; i < 8; ++i) {
                    double x_src_f = x_src_f_base + i * scale_width;
                    int x1 = static_cast<int>(floor(x_src_f));
                    int x2 = std::min(x1 + 1, input_image.cols - 1);
                    float dx = static_cast<float>(x_src_f - x1);
                    float one_minus_dx = 1.0f - dx;

                    // 对每个通道计算
                    for(int c = 0; c < 3; ++c) {
                        float p1 = input_image.at<Vec3f>(y1, x1)[c];
                        float p2 = input_image.at<Vec3f>(y1, x2)[c];
                        float p3 = input_image.at<Vec3f>(y2, x1)[c];
                        float p4 = input_image.at<Vec3f>(y2, x2)[c];

                        // 加载到Universal Intrinsics向量
                        // OpenCV的Universal Intrinsics API是较低层次的，通常不直接用于手工编写代码。
                        // 在这里，我们将示范如何使用OpenCV的Universal Intrinsics进行简单的运算。

                        // 使用Universal Intrinsics进行插值计算
                        v_float32x4 v_p1 = v_setall_f32(p1);
                        v_float32x4 v_p2 = v_setall_f32(p2);
                        v_float32x4 v_p3 = v_setall_f32(p3);
                        v_float32x4 v_p4 = v_setall_f32(p4);

                        v_float32x4 v_one_minus_dx = v_setall_f32(one_minus_dx);
                        v_float32x4 v_dx = v_setall_f32(dx);
                        v_float32x4 v_one_minus_dy = v_setall_f32(one_minus_dy_f);
                        v_float32x4 v_dy = v_setall_f32(dy);

                        // top = (1 - dx) * p1 + dx * p2
                        v_float32x4 top = v_fma(v_one_minus_dx, v_p1, v_mul(v_dx, v_p2));

                        // bottom = (1 - dx) * p3 + dx * p4
                        v_float32x4 bottom = v_fma(v_one_minus_dx, v_p3, v_mul(v_dx, v_p4));

                        // value = (1 - dy) * top + dy * bottom
                        v_float32x4 value = v_fma(v_one_minus_dy, top, v_mul(v_dy, bottom));

                        // 提取结果并存储
                        float final_val = v_reduce_sum(value);
                        output_image.at<Vec3f>(y_dst, x_dst + i)[c] = final_val;
                    }
                }
            }
        }
    });

    return output_image;
}

// 多线程任务结构体
struct ResizeTask {
    const Mat& input;
    Mat& output;
    double scale_width;
    double scale_height;
    string method; // "nearest" or "bilinear"

    ResizeTask(const Mat& in, Mat& out, double sw, double sh, const string& m)
        : input(in), output(out), scale_width(sw), scale_height(sh), method(m) {}
};

// 最近邻插值并行实现
template <typename T>
void nearestNeighborParallel(const ResizeTask& task, const Range& range) {
    for(int y_dst = range.start; y_dst < range.end; ++y_dst) {
        for(int x_dst = 0; x_dst < task.output.cols; ++x_dst) {
            double x_src_f = x_dst * task.scale_width;
            double y_src_f = y_dst * task.scale_height;
            T value;
            if(task.input.channels() == 1){
                value = nearestNeighbor<T>(task.input, x_src_f, y_src_f);
                task.output.at<T>(y_dst, x_dst) = value;
            }
            else if(task.input.channels() == 3){
                Vec<T,3> pixel = bilinearInterpolateVec<T>(task.input, x_src_f, y_src_f);
                task.output.at<Vec<T,3>>(y_dst, x_dst) = pixel;
            }
        }
    }
}

// 双线性插值并行实现
template <typename T>
void bilinearParallel(const ResizeTask& task, const Range& range) {
    for(int y_dst = range.start; y_dst < range.end; ++y_dst) {
        for(int x_dst = 0; x_dst < task.output.cols; ++x_dst) {
            double x_src_f = x_dst * task.scale_width;
            double y_src_f = y_dst * task.scale_height;
            if(task.input.channels() == 1){
                T value = bilinearInterpolate<T>(task.input, x_src_f, y_src_f);
                task.output.at<T>(y_dst, x_dst) = value;
            }
            else if(task.input.channels() == 3){
                Vec<T,3> value = bilinearInterpolateVec<T>(task.input, x_src_f, y_src_f);
                task.output.at<Vec3f>(y_dst, x_dst) = value;
            }
        }
    }
}

// 通用图像缩放函数（最近邻和双线性）
template <typename T>
Mat resizeImage(const Mat& input_image, int output_width, int output_height, const string& method) {
    Mat output_image(output_height, output_width, input_image.type());

    double scale_width = static_cast<double>(input_image.cols) / output_width;
    double scale_height = static_cast<double>(input_image.rows) / output_height;

    ResizeTask task(input_image, output_image, scale_width, scale_height, method);

    if(method == "nearest"){
        parallel_for_(Range(0, output_height), [&](const Range& range) {
            nearestNeighborParallel<T>(task, range);
        });
    }
    else if(method == "bilinear"){
        parallel_for_(Range(0, output_height), [&](const Range& range) {
            bilinearParallel<T>(task, range);
        });
    }

    return output_image;
}

// 主函数
int main(int argc, char* argv[]) {
    // 检查输入参数
    if(argc < 2){
        cout << "使用方法: " << argv[0] << " <input_image>" << endl;
        return -1;
    }

    string input_path = argv[1];

    // 读取输入图像
    Mat input_img = imread(input_path, IMREAD_UNCHANGED);
    if(input_img.empty()){
        cerr << "无法读取输入图像: " << input_path << endl;
        return -1;
    }

    // 打印图像信息
    cout << "输入图像大小: " << input_img.cols << "x" << input_img.rows << endl;
    cout << "图像类型: " << input_img.type() << " (channels: " << input_img.channels() << ")" << endl;

    // 定义输出尺寸（例如，缩放为原来的2倍）
    int output_width = input_img.cols * 2;
    int output_height = input_img.rows * 2;

    // 创建不同数据类型的版本
    Mat input_16UC1, input_16UC3, input_32FC1, input_32FC3;
    if(input_img.channels() == 1){
        input_img.convertTo(input_16UC1, CV_16UC1, 65535.0 / 255.0);
        input_img.convertTo(input_32FC1, CV_32FC1, 1.0 / 255.0);
    }
    else if(input_img.channels() == 3){
        input_img.convertTo(input_16UC3, CV_16UC3, 65535.0 / 255.0);
        input_img.convertTo(input_32FC3, CV_32FC3, 1.0 / 255.0);
    }

    // 开始计时
    auto start = high_resolution_clock::now();

    // 最近邻插值
    Mat resized_nn_16U, resized_nn_32F, resized_nn_16UC3, resized_nn_32FC3;
    if(input_img.channels() == 1){
        resized_nn_16U = resizeImage<ushort>(input_16UC1, output_width, output_height, "nearest");
        resized_nn_32F = resizeImage<float>(input_32FC1, output_width, output_height, "nearest");
    }
    else if(input_img.channels() == 3){
        resized_nn_16UC3 = resizeImage<ushort>(input_16UC3, output_width, output_height, "nearest");
        resized_nn_32FC3 = resizeImage<float>(input_32FC3, output_width, output_height, "nearest");
    }

    // 双线性插值
    Mat resized_bilinear_16U, resized_bilinear_32F, resized_bilinear_16UC3, resized_bilinear_32FC3;
    if(input_img.channels() == 1){
        resized_bilinear_16U = resizeImage<ushort>(input_16UC1, output_width, output_height, "bilinear");
        resized_bilinear_32F = resizeImage<float>(input_32FC1, output_width, output_height, "bilinear");
    }
    else if(input_img.channels() == 3){
        resized_bilinear_16UC3 = resizeImage<ushort>(input_16UC3, output_width, output_height, "bilinear");
        resized_bilinear_32FC3 = resizeImage<float>(input_32FC3, output_width, output_height, "bilinear");
    }

    // 使用AVX优化的双线性插值（仅CV_32FC3）
    Mat resized_bilinear_avx;
    if(input_img.channels() == 3){
        resized_bilinear_avx = bilinearResizeAVX(input_32FC3, output_width, output_height);
    }

    // 使用Universal Intrinsics优化的双线性插值（仅CV_32FC3）
    Mat resized_bilinear_universal;
    if(input_img.channels() == 3){
        resized_bilinear_universal = bilinearResizeUniversal(input_32FC3, output_width, output_height);
    }

    // 结束计时
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "图像缩放完成，耗时: " << duration << " 毫秒" << endl;

    // 保存结果
    if(input_img.channels() == 1){
        imwrite("resized_nn_16UC1.png", resized_nn_16U);
        imwrite("resized_nn_32FC1.png", resized_nn_32F);
        imwrite("resized_bilinear_16UC1.png", resized_bilinear_16U);
        imwrite("resized_bilinear_32FC1.png", resized_bilinear_32F);
    }
    else if(input_img.channels() == 3){
        imwrite("resized_nn_16UC3.png", resized_nn_16UC3);
        imwrite("resized_nn_32FC3.png", resized_nn_32FC3);
        imwrite("resized_bilinear_16UC3.png", resized_bilinear_16UC3);
        imwrite("resized_bilinear_32FC3.png", resized_bilinear_32FC3);
        imwrite("resized_bilinear_avx.png", resized_bilinear_avx);
        imwrite("resized_bilinear_universal.png", resized_bilinear_universal);
    }

    cout << "所有结果图像已保存。" << endl;

    return 0;
}
