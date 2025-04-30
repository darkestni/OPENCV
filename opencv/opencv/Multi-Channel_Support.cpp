#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <utility>

using namespace cv;
using namespace std;

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
void calculateScalingFactors(int input_width, int input_height, int output_width, int output_height, double &scale_width, double &scale_height) {
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
void assignPixelOptimized(const Mat& input_image, Mat& output_image, int x_dst, int y_dst, int x_src, int y_src) {
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
 * @brief 使用最近邻插值算法对图像进行缩放，支持单通道和三通道
 * 
 * @param input_image 输入的原始图像
 * @param output_width 期望的输出图像宽度
 * @param output_height 期望的输出图像高度
 * @return Mat 缩放后的图像
 */
Mat nearestNeighborResizeMultiChannel(const Mat& input_image, int output_width, int output_height) {
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

    for (int y_dst = 0; y_dst < output_height; ++y_dst) {
        for (int x_dst = 0; x_dst < output_width; ++x_dst) {
            // 反向映射到输入图像坐标
            pair<int, int> src_coords = getNearestNeighborCoordinates(x_dst, y_dst, scale_width, scale_height);
            int x_src = src_coords.first;
            int y_src = src_coords.second;

            // 限制坐标在有效范围内
            pair<int, int> clamped_coords = clampCoordinates(x_src, y_src, input_width, input_height);
            x_src = clamped_coords.first;
            y_src = clamped_coords.second;

            // 赋值像素值到输出图像（优化后的函数）
            assignPixelOptimized(input_image, output_image, x_dst, y_dst, x_src, y_src);
        }
    }

    return output_image;
}

/**
 * @brief 示例主函数，展示如何使用最近邻插值进行多通道图像的缩放
 */
int main() {
    // 读取输入图像（可以是灰度图像或彩色图像）
    Mat input_img = imread("input_image.jpg", IMREAD_UNCHANGED);
    if (input_img.empty()) {
        cout << "无法读取图像!" << endl;
        return -1;
    }

    // 打印图像信息
    cout << "输入图像尺寸: " << input_img.cols << "x" << input_img.rows << endl;
    cout << "图像通道数: " << input_img.channels() << endl;

    // 定义输出尺寸
    int new_width = 800;
    int new_height = 600;

    // 执行最近邻插值缩放，支持多通道
    Mat resized_img = nearestNeighborResizeMultiChannel(input_img, new_width, new_height);

    // 保存缩放后的图像
    imwrite("resized_image_multichannel.jpg", resized_img);

    // 可选：显示原始和缩放后的图像
    /*
    imshow("原始图像", input_img);
    imshow("缩放后图像 - 多通道支持", resized_img);
    waitKey(0);
    */

    cout << "图像缩放完成，保存到 resized_image_multichannel.jpg" << endl;
    return 0;
}

