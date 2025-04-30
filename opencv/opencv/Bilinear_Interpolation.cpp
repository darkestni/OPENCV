#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


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
/**
 * @brief 示例主函数，展示如何使用最近邻插值进行图像缩放
 */
int main() {
    // 读取输入图像
    Mat input_img = imread("input_image.jpg", IMREAD_UNCHANGED);
    if (input_img.empty()) {
        cout << "无法读取图像!" << endl;
        return -1;
    }
    // 定义输出尺寸
    int new_width = 800; // 可以调整为任意值，实现上采样或下采样
    int new_height = 600; // 可以调整为任意值，实现上采样或下采样

    // 执行双线性插值缩放
    Mat resized_img;
    BilinearInterpolationResizeBasic(input_img, resized_img, Size(new_width, new_height));

    // 保存缩放后的图像
    imwrite("resized_image_basic.jpg", resized_img);

    // 可选：显示原始和缩放后的图像
    /*
    imshow("原始图像", input_img);
    imshow("缩放后图像 - 基本实现", resized_img);
    waitKey(0);
    */

    cout << "图像缩放完成，保存到 resized_image_basic.jpg" << endl;
    return 0;
}
