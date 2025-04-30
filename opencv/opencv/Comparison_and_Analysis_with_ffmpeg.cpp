#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/core/parallel/parallel_backend.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace cv;
using namespace std;
using namespace std::chrono;

/**
 * @brief 计算图像的宽度和高度缩放因子
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
 * @brief 根据输出图像的坐标，通过缩放因子找到输入图像中最近的像素坐标
 *
 * @param x_dst 输出图像中的x坐标
 * @param y_dst 输出图像中的y坐标
 * @param scale_width 宽度缩放因子
 * @param scale_height 高度缩放因子
 * @return pair<int, int> 输入图像中最近的像素坐标 (x_src, y_src)
 */
pair<int, int> getNearestNeighborCoordinates(int x_dst, int y_dst, double scale_width, double scale_height) {
    // 计算输入图像中的对应浮点坐标
    double x_src_f = x_dst * scale_width;
    double y_src_f = y_dst * scale_height;

    // 四舍五入到最近的整数坐标
    int x_src = static_cast<int>(round(x_src_f));
    int y_src = static_cast<int>(round(y_src_f));

    return {x_src, y_src};
}

/**
 * @brief 将坐标限制在输入图像的有效范围内
 *
 * @param x_src 输入图像中的x坐标
 * @param y_src 输入图像中的y坐标
 * @param input_width 输入图像的宽度
 * @param input_height 输入图像的高度
 * @return pair<int, int> 限制后的坐标 (x_clamped, y_clamped)
 */
pair<int, int> clampCoordinates(int x_src, int y_src, int input_width, int input_height) {
    x_src = min(max(x_src, 0), input_width - 1);
    y_src = min(max(y_src, 0), input_height - 1);
    return {x_src, y_src};
}

/**
 * @brief 将输入图像中的像素值赋给输出图像，支持单通道和三通道
 *
 * @param input_image 输入图像
 * @param output_image 输出图像
 * @param x_dst 输出图像中的x坐标
 * @param y_dst 输出图像中的y坐标
 * @param x_src 输入图像中的x坐标
 * @param y_src 输入图像中的y坐标
 */
void assignPixelOptimized(const Mat &input_image, Mat &output_image, int x_dst, int y_dst, int x_src, int y_src) {
    int channels = input_image.channels();
    if (channels == 1) {
        // 单通道（灰度图像）
        output_image.at<uchar>(y_dst, x_dst) = input_image.at<uchar>(y_src, x_src);
    }
    else if (channels == 3) {
        // 三通道（例如RGB/BGR图像）
        Vec3b pixel = input_image.at<Vec3b>(y_src, x_src);
        output_image.at<Vec3b>(y_dst, x_dst) = pixel;
    }
    // 可以根据需要扩展更多通道的支持
}

/**
 * @brief 定义一个并行任务，用于在多线程环境下进行图像缩放
 */
struct ResizeTask : public ParallelLoopBody {
    const Mat &input_image;
    Mat &output_image;
    double scale_width;
    double scale_height;
    int output_height;
    int output_width;

    /**
     * @brief 构造函数
     *
     * @param in 输入图像
     * @param out 输出图像
     * @param sw 宽度缩放因子
     * @param sh 高度缩放因子
     * @param oh 输出图像高度
     * @param ow 输出图像宽度
     */
    ResizeTask(const Mat &in, Mat &out, double sw, double sh, int oh, int ow) :
        input_image(in), output_image(out), scale_width(sw), scale_height(sh), output_height(oh), output_width(ow) {}

    /**
     * @brief 重载运算符，用于并行处理指定范围的行
     *
     * @param range 需要处理的行范围
     */
    virtual void operator()(const Range &range) const CV_OVERRIDE {
        for (int y_dst = range.start; y_dst < range.end; ++y_dst) {
            for (int x_dst = 0; x_dst < output_width; ++x_dst) {
                // 反向映射到输入图像坐标
                pair<int, int> src_coords = getNearestNeighborCoordinates(x_dst, y_dst, scale_width, scale_height);
                int x_src = src_coords.first;
                int y_src = src_coords.second;

                // 限制坐标在有效范围内
                pair<int, int> clamped_coords = clampCoordinates(x_src, y_src, input_image.cols, input_image.rows);
                x_src = clamped_coords.first;
                y_src = clamped_coords.second;

                // 赋值像素值到输出图像
                assignPixelOptimized(input_image, output_image, x_dst, y_dst, x_src, y_src);
            }
        }
    }
};

/**
 * @brief 使用最近邻插值算法对图像进行缩放，支持多线程处理以优化性能
 *
 * @param input_image 输入的原始图像
 * @param output_width 期望的输出图像宽度
 * @param output_height 期望的输出图像高度
 * @return Mat 缩放后的图像
 */
Mat nearestNeighborResizeParallel(const Mat &input_image, int output_width, int output_height) {
    int input_width = input_image.cols;
    int input_height = input_image.rows;

    double scale_width, scale_height;
    calculateScalingFactors(input_width, input_height, output_width, output_height, scale_width, scale_height);

    // 初始化输出图像
    Mat output_image;
    if (input_image.channels() == 1) {
        output_image = Mat::zeros(output_height, output_width, CV_8UC1);
    }
    else {
        output_image = Mat::zeros(output_height, output_width, input_image.type());
    }

    // 创建并行任务
    ResizeTask task(input_image, output_image, scale_width, scale_height, output_height, output_width);

    // 使用 parallel_for_ 进行多线程处理
    parallel_for_(Range(0, output_height), task);

    return output_image;
}

/**
 * @brief 使用OpenCV的resize函数进行最近邻插值缩放
 *
 * @param input_image 输入的原始图像
 * @param output_width 期望的输出图像宽度
 * @param output_height 期望的输出图像高度
 * @return Mat 缩放后的图像
 */
Mat opencvResize(const Mat &input_image, int output_width, int output_height) {
    Mat resized_image;
    resize(input_image, resized_image, Size(output_width, output_height), 0, 0, INTER_NEAREST);
    return resized_image;
}

/**
 * @brief 使用FFmpeg进行图像缩放
 */
Mat ffmpegResize(const Mat &input_image, int output_width, int output_height) {
    // 创建SwsContext
    SwsContext *sws_ctx = sws_getContext(input_image.cols, input_image.rows, AV_PIX_FMT_BGR24, // 源格式
                                         output_width, output_height, AV_PIX_FMT_BGR24, // 目标格式
                                         SWS_POINT, // 最近邻插值
                                         nullptr, nullptr, nullptr);

    if (!sws_ctx) {
        throw runtime_error("无法创建SwsContext");
    }

    // 准备输入数据
    uint8_t *src_data[4] = {input_image.data, nullptr, nullptr, nullptr};
    int src_linesize[4] = {static_cast<int>(input_image.step), 0, 0, 0};

    // 准备输出数据
    Mat output_image(output_height, output_width, input_image.type());
    uint8_t *dst_data[4] = {output_image.data, nullptr, nullptr, nullptr};
    int dst_linesize[4] = {static_cast<int>(output_image.step), 0, 0, 0};

    // 执行缩放
    sws_scale(sws_ctx, src_data, src_linesize, 0, input_image.rows, dst_data, dst_linesize);

    // 清理
    sws_freeContext(sws_ctx);

    return output_image;
}

/**
 * @brief 双线性插值缩放函数
 *
 * @param input 输入图像
 * @param output 输出图像
 * @param new_size 期望的输出图像尺寸
 */
void BilinearInterpolationResizeBasic(const Mat &input, Mat &output, const Size &new_size) {
    double scale_width = static_cast<double>(input.cols) / new_size.width;
    double scale_height = static_cast<double>(input.rows) / new_size.height;

    output = Mat::zeros(new_size, input.type());

    for (int i = 0; i < new_size.width; ++i) {
        for (int j = 0; j < new_size.height; ++j) {
            double x_src = (i + 0.5) * scale_width - 0.5;
            double y_src = (j + 0.5) * scale_height - 0.5;
            int x_1 = static_cast<int>(x_src);
            int y_1 = static_cast<int>(y_src);
            int x_2 = min(x_1 + 1, input.cols - 1);
            int y_2 = min(y_1 + 1, input.rows - 1);

            double weight_x_2 = x_src - x_1;
            double weight_x_1 = 1 - weight_x_2;
            double weight_y_2 = y_src - y_1;
            double weight_y_1 = 1 - weight_y_2;

            if (input.type() == CV_8UC1) {
                double p1 = input.at<uchar>(y_1, x_1);
                double p2 = input.at<uchar>(y_1, x_2);
                double p3 = input.at<uchar>(y_2, x_1);
                double p4 = input.at<uchar>(y_2, x_2);

                double p_x_1 = weight_x_1 * p1 + weight_x_2 * p2;
                double p_x_2 = weight_x_1 * p3 + weight_x_2 * p4;
                double p_dst = weight_y_1 * p_x_1 + weight_y_2 * p_x_2;

                output.at<uchar>(j, i) = static_cast<uchar>(p_dst);
            }
            else if (input.type() == CV_8UC3) {
                Vec3b p1 = input.at<Vec3b>(y_1, x_1);
                Vec3b p2 = input.at<Vec3b>(y_1, x_2);
                Vec3b p3 = input.at<Vec3b>(y_2, x_1);
                Vec3b p4 = input.at<Vec3b>(y_2, x_2);

                Vec3b p_x_1 = weight_x_1 * p1 + weight_x_2 * p2;
                Vec3b p_x_2 = weight_x_1 * p3 + weight_x_2 * p4;
                Vec3b p_dst = weight_y_1 * p_x_1 + weight_y_2 * p_x_2;

                output.at<Vec3b>(j, i) = p_dst;
            }
        }
    }
}
Mat bilinearResize(const Mat &input_image, int output_width, int output_height) {
    Mat output_image;
    BilinearInterpolationResizeBasic(input_image, output_image, Size(output_width, output_height));
    return output_image;
}

// 添加 OpenCV 双线性插值函数
Mat opencvBilinearResize(const Mat &input_image, int output_width, int output_height) {
    Mat resized_image;
    resize(input_image, resized_image, Size(output_width, output_height), 0, 0, INTER_LINEAR);
    return resized_image;
}

// 添加 FFmpeg 双线性插值函数
Mat ffmpegBilinearResize(const Mat &input_image, int output_width, int output_height) {
    SwsContext *sws_ctx = sws_getContext(input_image.cols, input_image.rows, AV_PIX_FMT_BGR24, output_width,
                                         output_height, AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, // 使用双线性插值
                                         nullptr, nullptr, nullptr);

    if (!sws_ctx) {
        throw runtime_error("无法创建SwsContext");
    }

    uint8_t *src_data[4] = {input_image.data, nullptr, nullptr, nullptr};
    int src_linesize[4] = {static_cast<int>(input_image.step), 0, 0, 0};

    Mat output_image(output_height, output_width, input_image.type());
    uint8_t *dst_data[4] = {output_image.data, nullptr, nullptr, nullptr};
    int dst_linesize[4] = {static_cast<int>(output_image.step), 0, 0, 0};

    sws_scale(sws_ctx, src_data, src_linesize, 0, input_image.rows, dst_data, dst_linesize);
    sws_freeContext(sws_ctx);

    return output_image;
}

/**
 * @brief 主函数，测试并比较自实现的多线程最近邻缩放与OpenCV的性能
 */
int main(int argc, char **argv) {
    // 检查命令行参数
    if (argc < 5) {
        cout << "使用方法: " << argv[0] << " <输入图像路径> <自定义输出图像路径> <输出宽度> <输出高度>" << endl;
        return -1;
    }

    string input_path = argv[1];
    string output_path_custom = argv[2];
    int new_width = stoi(argv[3]);
    int new_height = stoi(argv[4]);

    // 读取输入图像
    Mat input_img = imread(input_path, IMREAD_UNCHANGED);
    if (input_img.empty()) {
        cout << "无法读取图像: " << input_path << endl;
        return -1;
    }

    // 打印图像信息
    cout << "输入图像尺寸: " << input_img.cols << "x" << input_img.rows << endl;
    cout << "图像通道数: " << input_img.channels() << endl;

    // 定义测试参数
    struct TestCase {
        string name;
        int output_width;
        int output_height;
    };

    vector<TestCase> test_cases = {{"Test1_Upscale", input_img.cols * 2, input_img.rows * 2},
                                   {"Test2_Downscale", input_img.cols / 2, input_img.rows / 2},
                                   {"Test3_AspectRatioChange", input_img.cols * 3 / 4, input_img.rows * 5 / 4}};

    // 测试次数以获得平均值
    int test_iterations = 10;

    // 在main函数的测试循环中添加：
    for (const auto &test : test_cases) {
        cout << "\n=== " << test.name << " ===" << endl;
        cout << "输出尺寸: " << test.output_width << "x" << test.output_height << endl;

        // 测试自实现的多线程最近邻缩放
        auto start_custom = high_resolution_clock::now();
        Mat resized_img_custom;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_custom = nearestNeighborResizeParallel(input_img, test.output_width, test.output_height);
        }
        auto end_custom = high_resolution_clock::now();
        auto duration_custom = duration_cast<milliseconds>(end_custom - start_custom).count();
        double avg_time_custom = static_cast<double>(duration_custom) / test_iterations;

        // 测试 OpenCV 的 最近邻缩放
        auto start_opencv = high_resolution_clock::now();
        Mat resized_img_opencv;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_opencv = opencvResize(input_img, test.output_width, test.output_height);
        }
        auto end_opencv = high_resolution_clock::now();
        auto duration_opencv = duration_cast<milliseconds>(end_opencv - start_opencv).count();
        double avg_time_opencv = static_cast<double>(duration_opencv) / test_iterations;

        // 测试 FFmpeg 的最近邻缩放
        auto start_ffmpeg = high_resolution_clock::now();
        Mat resized_img_ffmpeg;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_ffmpeg = ffmpegResize(input_img, test.output_width, test.output_height);
        }
        auto end_ffmpeg = high_resolution_clock::now();
        auto duration_ffmpeg = duration_cast<milliseconds>(end_ffmpeg - start_ffmpeg).count();
        double avg_time_ffmpeg = static_cast<double>(duration_ffmpeg) / test_iterations;

        // 测试自实现双线性插值缩放
        auto start_bilinear = high_resolution_clock::now();
        Mat resized_img_bilinear;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_bilinear = bilinearResize(input_img, test.output_width, test.output_height);
        }
        auto end_bilinear = high_resolution_clock::now();
        auto duration_bilinear = duration_cast<milliseconds>(end_bilinear - start_bilinear).count();
        double avg_time_bilinear = static_cast<double>(duration_bilinear) / test_iterations;

        // 测试 OpenCV 的双线性插值缩放
        auto start_opencv_bilinear = high_resolution_clock::now();
        Mat resized_img_opencv_bilinear;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_opencv_bilinear = opencvBilinearResize(input_img, test.output_width, test.output_height);
        }
        auto end_opencv_bilinear = high_resolution_clock::now();
        auto duration_opencv_bilinear =
            duration_cast<milliseconds>(end_opencv_bilinear - start_opencv_bilinear).count();
        double avg_time_opencv_bilinear = static_cast<double>(duration_opencv_bilinear) / test_iterations;

        // 测试 FFmpeg 的双线性插值缩放
        auto start_ffmpeg_bilinear = high_resolution_clock::now();
        Mat resized_img_ffmpeg_bilinear;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_ffmpeg_bilinear = ffmpegBilinearResize(input_img, test.output_width, test.output_height);
        }
        auto end_ffmpeg_bilinear = high_resolution_clock::now();
        auto duration_ffmpeg_bilinear =
            duration_cast<milliseconds>(end_ffmpeg_bilinear - start_ffmpeg_bilinear).count();
        double avg_time_ffmpeg_bilinear = static_cast<double>(duration_ffmpeg_bilinear) / test_iterations;

        // 保存结果图像
        string custom_output = "custom_" + test.name + ".jpg";
        string opencv_output = "opencv_" + test.name + ".jpg";
        string ffmpeg_output = "ffmpeg_" + test.name + ".jpg";
        string bilinear_output = "bilinear_" + test.name + ".jpg";
        string opencv_bilinear_output = "opencv_bilinear_" + test.name + ".jpg";
        string ffmpeg_bilinear_output = "ffmpeg_bilinear_" + test.name + ".jpg";

        imwrite(custom_output, resized_img_custom);
        imwrite(opencv_output, resized_img_opencv);
        imwrite(ffmpeg_output, resized_img_ffmpeg);
        imwrite(bilinear_output, resized_img_bilinear);
        imwrite(opencv_bilinear_output, resized_img_opencv_bilinear);
        imwrite(ffmpeg_bilinear_output, resized_img_ffmpeg_bilinear);

        // 输出比较结果
        cout << "自实现多线程最近邻缩放平均时间: " << avg_time_custom << " ms" << endl;
        cout << "OpenCV 最近邻缩放平均时间: " << avg_time_opencv << " ms" << endl;
        cout << "FFmpeg 最近邻缩放平均时间: " << avg_time_ffmpeg << " ms" << endl;
        cout << "自实现双线性插值缩放平均时间: " << avg_time_bilinear << " ms" << endl;
        cout << "OpenCV 双线性插值缩放平均时间: " << avg_time_opencv_bilinear << " ms" << endl;
        cout << "FFmpeg 双线性插值缩放平均时间: " << avg_time_ffmpeg_bilinear << " ms" << endl;
        
        // 计算PSNR并输出图像质量比较
        double psnr_custom = PSNR(resized_img_opencv, resized_img_custom);
        double psnr_ffmpeg = PSNR(resized_img_opencv, resized_img_ffmpeg);
        double psnr_bilinear = PSNR(resized_img_opencv, resized_img_bilinear);
        double psnr_opencv_bilinear = PSNR(resized_img_opencv, resized_img_opencv_bilinear);
        double psnr_ffmpeg_bilinear = PSNR(resized_img_opencv, resized_img_ffmpeg_bilinear);
        
        cout << "\n图像质量比较 (PSNR，以OpenCV为基准):" << endl;
        cout << "自实现最近邻 PSNR: " << psnr_custom << " dB" << endl;
        cout << "FFmpeg PSNR: " << psnr_ffmpeg << " dB" << endl;
        cout << "自实现双线性插值 PSNR: " << psnr_bilinear << " dB" << endl;
        cout << "OpenCV 双线性插值 PSNR: " << psnr_opencv_bilinear << " dB" << endl;
        cout << "FFmpeg 双线性插值 PSNR: " << psnr_ffmpeg_bilinear << " dB" << endl;
    }

    return 0;
}

