#include <opencv2/opencv.hpp>
#include <opencv2/core/parallel/parallel_backend.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <immintrin.h>
#include <chrono>
#include <stdexcept>
#include <cmath>

using namespace cv;
using namespace std;
using namespace std::chrono;

/**
 * @brief 计算图像的缩放因子
 *
 * @param input_width 输入图像的宽度
 * @param input_height 输入图像的高度
 * @param output_width 输出图像的宽度
 * @param output_height 输出图像的高度
 * @param scale_width 输出图像宽度的缩放因子
 * @param scale_height 输出图像高度的缩放因子
 */
void calculateScalingFactors(int input_width, int input_height, int output_width, int output_height,
                             double &scale_width, double &scale_height) {
    scale_width = static_cast<double>(input_width) / output_width;
    scale_height = static_cast<double>(input_height) / output_height;
}

/**
 * @brief 获取最近邻插值的源坐标
 *
 * @param x_dst 输出图像中的x坐标
 * @param y_dst 输出图像中的y坐标
 * @param scale_width 宽度缩放因子
 * @param scale_height 高度缩放因子
 * @return pair<int, int> 输入图像中的最近邻坐标 (x_src, y_src)
 */
pair<int, int> getNearestNeighborCoordinates(int x_dst, int y_dst, double scale_width, double scale_height) {
    double x_src_f = (x_dst + 0.5) * scale_width - 0.5;
    double y_src_f = (y_dst + 0.5) * scale_height - 0.5;
    int x_src = static_cast<int>(round(x_src_f));
    int y_src = static_cast<int>(round(y_src_f));
    return {x_src, y_src};
}

/**
 * @brief 限制坐标在有效范围内
 *
 * @param x_src 输入图像中的x坐标
 * @param y_src 输入图像中的y坐标
 * @param input_width 输入图像的宽度
 * @param input_height 输入图像的高度
 * @return pair<int, int> 被限制后的坐标
 */
pair<int, int> clampCoordinates(int x_src, int y_src, int input_width, int input_height) {
    x_src = min(max(x_src, 0), input_width - 1);
    y_src = min(max(y_src, 0), input_height - 1);
    return {x_src, y_src};
}

/**
 * @brief 赋值像素，支持多种数据类型和通道数
 *
 * @tparam T 数据类型
 * @param input_image 输入图像
 * @param output_image 输出图像
 * @param x_dst 输出图像中的x坐标
 * @param y_dst 输出图像中的y坐标
 * @param x_src 输入图像中的x坐标
 * @param y_src 输入图像中的y坐标
 */
template <typename T>
void assignPixelOptimized(const Mat &input_image, Mat &output_image, int x_dst, int y_dst, int x_src, int y_src) {
    int channels = input_image.channels();
    if (channels == 1) {
        // 单通道
        output_image.at<T>(y_dst, x_dst) = input_image.at<T>(y_src, x_src);
    }
    else if (channels == 3) {
        // 三通道
        Vec<T, 3> pixel = input_image.at<Vec<T, 3>>(y_src, x_src);
        output_image.at<Vec<T, 3>>(y_dst, x_dst) = pixel;
    }
    // 可扩展更多通道
}

/**
 * @brief 最近邻插值的多线程任务
 */
struct NearestNeighborTask : public cv::ParallelLoopBody {
    const Mat &input_image;
    Mat &output_image;
    double scale_width;
    double scale_height;
    int output_width;
    int output_height;

    NearestNeighborTask(const Mat &in, Mat &out, double sw, double sh, int ow, int oh)
        : input_image(in), output_image(out), scale_width(sw), scale_height(sh),
          output_width(ow), output_height(oh) {}

    void operator()(const Range& range) const CV_OVERRIDE {
        int channels = input_image.channels();
        for (int y_dst = range.start; y_dst < range.end; ++y_dst) {
            for (int x_dst = 0; x_dst < output_width; ++x_dst) {
                pair<int, int> src_coords = getNearestNeighborCoordinates(x_dst, y_dst, scale_width, scale_height);
                pair<int, int> clamped_coords = clampCoordinates(src_coords.first, src_coords.second, input_image.cols, input_image.rows);
                // 根据数据类型选择对应的模板实例
                if (input_image.depth() == CV_8U) {
                    assignPixelOptimized<uchar>(input_image, output_image, x_dst, y_dst, clamped_coords.first, clamped_coords.second);
                }
                else if (input_image.depth() == CV_16U) {
                    assignPixelOptimized<ushort>(input_image, output_image, x_dst, y_dst, clamped_coords.first, clamped_coords.second);
                }
                else if (input_image.depth() == CV_32F) {
                    assignPixelOptimized<float>(input_image, output_image, x_dst, y_dst, clamped_coords.first, clamped_coords.second);
                }
                // 可扩展更多数据类型
            }
        }
    }
};

/**
 * @brief 最近邻插值函数，支持多数据类型和多线程
 *
 * @param input_image 输入图像
 * @param output_width 输出图像的宽度
 * @param output_height 输出图像的高度
 * @return Mat 缩放后的图像
 */
Mat nearestNeighborResizeParallel(const Mat &input_image, int output_width, int output_height) {
    // 计算缩放因子
    double scale_width, scale_height;
    calculateScalingFactors(input_image.cols, input_image.rows, output_width, output_height, scale_width, scale_height);
    
    // 初始化输出图像
    Mat output_image = Mat::zeros(output_height, output_width, input_image.type());

    // 创建并行任务并执行多线程缩放
    NearestNeighborTask task(input_image, output_image, scale_width, scale_height, output_width, output_height);
    parallel_for_(Range(0, output_height), task);

    return output_image;
}

/**
 * @brief 双线性插值的基础实现
 *
 * @tparam T 数据类型
 * @param input 输入图像
 * @param output 输出图像
 * @param new_size 输出图像大小
 */
template <typename T>
void bilinearInterpolationResizeBasic(const Mat &input, Mat &output, const Size &new_size) {
    double scale_width = static_cast<double>(input.cols) / new_size.width;
    double scale_height = static_cast<double>(input.rows) / new_size.height;
    
    int channels = input.channels();
    for (int y_dst = 0; y_dst < new_size.height; ++y_dst) {
        double y_src_f = (y_dst + 0.5) * scale_height - 0.5;
        int y_src = static_cast<int>(floor(y_src_f));
        double dy = y_src_f - y_src;

        y_src = min(max(y_src, 0), input.rows - 2);
        double dy1 = 1.0 - dy;

        for (int x_dst = 0; x_dst < new_size.width; ++x_dst) {
            double x_src_f = (x_dst + 0.5) * scale_width - 0.5;
            int x_src = static_cast<int>(floor(x_src_f));
            double dx = x_src_f - x_src;

            x_src = min(max(x_src, 0), input.cols - 2);
            double dx1 = 1.0 - dx;
            
            for (int c = 0; c < channels; ++c) {
                T p1, p2, p3, p4;

                if (channels == 1) {
                    p1 = input.at<T>(y_src, x_src);
                    p2 = input.at<T>(y_src, x_src + 1);
                    p3 = input.at<T>(y_src + 1, x_src);
                    p4 = input.at<T>(y_src + 1, x_src + 1);
                }
                else if (channels == 3) {
                    Vec<T, 3> vec1 = input.at<Vec<T, 3>>(y_src, x_src);
                    Vec<T, 3> vec2 = input.at<Vec<T, 3>>(y_src, x_src + 1);
                    Vec<T, 3> vec3 = input.at<Vec<T, 3>>(y_src + 1, x_src);
                    Vec<T, 3> vec4 = input.at<Vec<T, 3>>(y_src + 1, x_src + 1);
                    p1 = vec1[c];
                    p2 = vec2[c];
                    p3 = vec3[c];
                    p4 = vec4[c];
                }

                double interp1 = p1 * dx1 + p2 * dx;
                double interp2 = p3 * dx1 + p4 * dx;
                double interp_final = interp1 * dy1 + interp2 * dy;

                if (channels == 1) {
                    output.at<T>(y_dst, x_dst) = static_cast<T>(interp_final);
                }
                else if (channels == 3) {
                    output.at<Vec<T, 3>>(y_dst, x_dst)[c] = static_cast<T>(interp_final);
                }
            }
        }
    }
}

/**
 * @brief 双线性插值的 AVX 加速实现，仅支持 CV_32FC1 和 CV_32FC3
 *
 * @param input 输入图像
 * @param output 输出图像
 * @param new_size 输出图像大小
 */
void bilinearResizeAVX(const Mat &input, Mat &output, const Size &new_size) {
    double scale_width = static_cast<double>(input.cols) / new_size.width;
    double scale_height = static_cast<double>(input.rows) / new_size.height;

    int channels = input.channels();
    output = Mat::zeros(new_size, input.type());

    // 仅支持 CV_32F 数据类型
    if (input.depth() != CV_32F) {
        throw runtime_error("AVX加速仅支持 CV_32FC1 和 CV_32FC3 数据类型");
    }

    for (int y_dst = 0; y_dst < new_size.height; ++y_dst) {
        float y_src_f = static_cast<float>((y_dst + 0.5) * scale_height - 0.5);
        int y_src = static_cast<int>(floor(y_src_f));
        float dy = y_src_f - y_src;

        y_src = min(max(y_src, 0), input.rows - 2);
        float dy1 = 1.0f - dy;
        __m256 v_dy1 = _256_set1_ps(dy1);
        __m256 v_dy = _mm256_set1_ps(dy);

        for (int x_dst = 0; x_dst < new_size.width; x_dst += 8) { // 每次处理8个像素
            int remaining = new_size.width - x_dst;
            int batch = remaining >= 8 ? 8 : remaining;

            // 生成x坐标向量
            float x_coords[8];
            for (int i = 0; i < 8; ++i) {
                if (x_dst + i < new_size.width) {
                    x_coords[i] = static_cast<float>((x_dst + i + 0.5) * scale_width - 0.5);
                }
                else {
                    x_coords[i] = 0.0f; // 不使用的元素设置为0
                }
            }
            __m256 v_x_src_f = _mm256_loadu_ps(x_coords);
            __m256 v_x1_f = _mm256_floor_ps(v_x_src_f);
            __m256 v_dx = _mm256_sub_ps(v_x_src_f, v_x1_f);
            __m256 v_dx1 = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_dx);

            __m256i v_x1 = _mm256_cvtps_epi32(v_x1_f);
            __m256i v_x2 = _mm256_add_epi32(v_x1, _mm256_set1_epi32(1));

            // 裁剪坐标
            __m256i zero = _mm256_set1_epi32(0);
            __m256i max_x = _mm256_set1_epi32(input.cols - 1);
            v_x1 = _mm256_min_epi32(_mm256_max_epi32(v_x1, zero), max_x);
            v_x2 = _mm256_min_epi32(_mm256_max_epi32(v_x2, zero), max_x);

            for (int c = 0; c < channels; ++c) {
                // 加载四个邻近像素
                float *row_ptr1 = input.ptr<float>(y_src) + c;
                float *row_ptr2 = input.ptr<float>(y_src + 1) + c;

                // 使用 gather 指令加载像素值
                __m256 p1 = _mm256_i32gather_ps(row_ptr1, v_x1, sizeof(float));
                __m256 p2 = _mm256_i32gather_ps(row_ptr1 + channels, v_x2, sizeof(float));
                __m256 p3 = _mm256_i32gather_ps(row_ptr2, v_x1, sizeof(float));
                __m256 p4 = _mm256_i32gather_ps(row_ptr2 + channels, v_x2, sizeof(float));

                // 计算插值
                __m256 interp1 = _mm256_mul_ps(_mm256_set1_ps(1.0f), p1); // 可根据需要调整权重
                __m256 interp2 = _mm256_mul_ps(v_dx, p2);
                __m256 interp3 = _mm256_mul_ps(_mm256_set1_ps(1.0f), p3);
                __m256 interp4 = _mm256_mul_ps(v_dx, p4);

                __m256 interp_final1 = _mm256_add_ps(interp1, interp2);
                __m256 interp_final2 = _mm256_add_ps(interp3, interp4);
                __m256 result = _mm256_add_ps(_mm256_mul_ps(interp_final1, v_dy1), _mm256_mul_ps(interp_final2, v_dy));

                // 存储结果
                float output_vals[8];
                _mm256_storeu_ps(output_vals, result);

                for (int i = 0; i < batch; ++i) {
                    if (x_dst + i < new_size.width) {
                        if (channels == 1) {
                            output.at<float>(y_dst, x_dst + i) = output_vals[i];
                        }
                        else if (channels == 3) {
                            Vec<float, 3> &pixel = output.at<Vec<float, 3>>(y_dst, x_dst + i);
                            pixel[c] = output_vals[i];
                        }
                    }
                }
            }
        }
    }

/**
 * @brief 通用双线性插值函数，自动选择数据类型和优化策略
 *
 * @param input_image 输入图像
 * @param output_width 输出图像的宽度
 * @param output_height 输出图像的高度
 * @return Mat 缩放后的图像
 */
Mat bilinearResize(const Mat &input_image, int output_width, int output_height) {
    Mat output_image;
    Size new_size(output_width, output_height);

    if (input_image.depth() == CV_8U || input_image.depth() == CV_16U || input_image.depth() == CV_32F) {
        if (input_image.depth() == CV_8U) {
            bilinearInterpolationResizeBasic<uchar>(input_image, output_image, new_size);
        }
        else if (input_image.depth() == CV_16U) {
            bilinearInterpolationResizeBasic<ushort>(input_image, output_image, new_size);
        }
        else if (input_image.depth() == CV_32F) {
            try {
                bilinearResizeAVX(input_image, output_image, new_size);
            }
            catch (const runtime_error &e) {
                cerr << "AVX优化失败: " << e.what() << endl;
                bilinearInterpolationResizeBasic<float>(input_image, output_image, new_size);
            }
        }
    }
    else {
        throw runtime_error("不支持的数据类型");
    }

    return output_image;
}

int main(int argc, char **argv) {
    // 检查命令行参数，需有5个参数：输入图像路径、输出最近邻前缀、输出双线性前缀、输出宽度、输出高度
    if (argc < 6) {
        cout << "使用方法: " << argv[0] << " <输入图像路径> <输出图像路径_最近邻前缀> <输出图像路径_双线性前缀> <输出宽度> <输出高度>" << endl;
        return -1;
    }

    string input_path = argv[1];
    string output_path_nn = argv[2];
    string output_path_bilinear = argv[3];
    int new_width = stoi(argv[4]);
    int new_height = stoi(argv[5]);

    // 读取输入图像，保持原始数据类型
    Mat input_img = imread(input_path, IMREAD_UNCHANGED);
    if (input_img.empty()) {
        cout << "无法读取图像: " << input_path << endl;
        return -1;
    }

    // 打印图像信息
    cout << "输入图像尺寸: " << input_img.cols << "x" << input_img.rows << endl;
    cout << "图像通道数: " << input_img.channels() << endl;
    cout << "图像数据类型: ";
    switch (input_img.depth()) {
        case CV_8U: cout << "CV_8U"; break;
        case CV_16U: cout << "CV_16U"; break;
        case CV_32F: cout << "CV_32F"; break;
        default: cout << "其他类型"; break;
    }
    cout << endl;

    // 定义测试参数
    struct TestCase {
        string name;
        int output_width;
        int output_height;
    };

    vector<TestCase> test_cases = {
        {"Upscale", new_width, new_height},
        {"Downscale", new_width / 2, new_height / 2},
        {"AspectRatioChange", static_cast<int>(new_width * 0.75), static_cast<int>(new_height * 1.25)}
    };

    // 测试次数以获得平均值
    int test_iterations = 10;

    for (const auto &test : test_cases) {
        cout << "\n=== " << test.name << " ===" << endl;
        cout << "输出尺寸: " << test.output_width << "x" << test.output_height << endl;

        // 测试最近邻缩放
        auto start_nn = high_resolution_clock::now();
        Mat resized_nn;
        for (int i = 0; i < test_iterations; ++i) {
            resized_nn = nearestNeighborResizeParallel(input_img, test.output_width, test.output_height);
        }
        auto end_nn = high_resolution_clock::now();
        auto duration_nn = duration_cast<milliseconds>(end_nn - start_nn).count();
        double avg_time_nn = static_cast<double>(duration_nn) / test_iterations;
        cout << "最近邻缩放平均时间: " << avg_time_nn << " ms" << endl;

        // 保存最近邻缩放结果
        string nn_output = output_path_nn + "_" + test.name + ".png";
        imwrite(nn_output, resized_nn);

        // 测试双线性插值缩放
        auto start_bilinear = high_resolution_clock::now();
        Mat resized_bilinear;
        for (int i = 0; i < test_iterations; ++i) {
            resized_bilinear = bilinearResize(input_img, test.output_width, test.output_height);
        }
        auto end_bilinear = high_resolution_clock::now();
        auto duration_bilinear = duration_cast<milliseconds>(end_bilinear - start_bilinear).count();
        double avg_time_bilinear = static_cast<double>(duration_bilinear) / test_iterations;
        cout << "双线性插值缩放平均时间: " << avg_time_bilinear << " ms" << endl;

        // 保存双线性插值缩放结果
        string bilinear_output = output_path_bilinear + "_" + test.name + ".png";
        imwrite(bilinear_output, resized_bilinear);
    }

    return 0;
}
