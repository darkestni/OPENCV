最近邻图像缩放项目

概述
本项目使用 C++ 和 OpenCV 实现了 最近邻插值 算法，用于图像缩放。项目涵盖了基本的图像缩放功能，并扩展支持多通道处理、任意缩放因子以及通过多线程优化性能。此外，项目还包含了自定义实现与
OpenCV 内置函数的性能对比分析。

功能

1.基本最近邻插值：使用最近邻方法对灰度图像和 RGB/BGR 彩色图像进行缩放。
2.多通道支持：无缝处理单通道（灰度）和多通道（彩色）图像。
3.任意缩放因子：支持上采样、下采样以及不同比例的宽高比变化，实现任意尺寸的图像缩放。
4.多线程优化：利用多线程技术提升缩放算法的性能，缩短处理时间。
5.性能对比：对比自定义实现与 OpenCV 内置 `resize` 函数的效率和速度，分析两者的性能差异。

一.前提
1.C++ 编译器：确保系统中已安装 C++ 编译器（例如 `g++`）。
2.OpenCV 库：在系统中安装 OpenCV（版本 3 或 4）。

二.编译

使用 `g++` 编译器和 `pkg-config` 来链接 OpenCV 库。以下是项目各组件的编译。

2.1基本最近邻实现

bash
g++ nearest_neighbor_basic.cpp -o nearest_neighbor_basic `pkg-config --cflags --libs opencv4`

多通道支持

bash
g++ nearest_neighbor_multichannel.cpp -o nearest_neighbor_multichannel `pkg-config --cflags --libs opencv4`

任意缩放因子

bash
g++ nearest_neighbor_arbitrary.cpp -o nearest_neighbor_arbitrary `pkg-config --cflags --libs opencv4`

多线程优化与性能对比

bash
g++ performance_comparison.cpp -o performance_comparison `pkg-config --cflags --libs opencv4`

双线性插值
bash
```
g++ bilinear_interpolation.cpp -o bilinear_interpolation `pkg-config --cflags --libs opencv4`
```

三.使用

1.基本最近邻缩放

1.1 运行可执行文件

./nearest_neighbor_basic

1.2 行为

- 从当前目录读取 `input_image.jpg`。
- 将图像缩放到预定义尺寸（例如，800x600）。
- 将缩放后的图像保存为 `resized_image_basic.jpg`。

2. 多通道支持

2.1 运行可执行文件
bash
./nearest_neighbor_multichannel

2.2 行为

- 读取彩色或灰度图像。
- 适当处理多通道数据。
- 将缩放后的图像保存为 `resized_image_multichannel.jpg`。

3.任意缩放因子
3.1 运行可执行文件

   ```bash
   ./nearest_neighbor_arbitrary
   ```

3.2. 行为

- 使用非整数缩放因子调整图像尺寸。
- 支持宽高比的任意变化。
- 根据需要保存缩放后的图像。

4.性能对比

4.1运行可执行文件并提供命令行参数

   ```bash
   ./performance_comparison <输入图像路径> <自定义输出路径> <输出宽度> <输出高度>
  ```

示例：

   ```bash
   ./performance_comparison input_image.jpg custom_resized.jpg 800 600
  ```

4.2. 行为

- 使用自定义的多线程缩放实现和 OpenCV 的 `resize` 函数分别进行多次缩放操作。
- 记录并显示每种方法的平均处理时间。
- 将缩放后的图像保存为不同的文件名，便于比较。

5.测试

为了确保实现的正确性：

5.1. 准备测试图像

- 使用不同尺寸和通道数（灰度和彩色）的图像进行测试。

5.2. 运行各组件
-执行每个编译后的程序，验证输出图像是否按预期缩放。

5.3. 性能分析

- 使用性能对比工具评估自定义实现与 OpenCV 内置函数在不同缩放场景下的效率和速度。

6. 性能分析

项目包含一个性能基准测试工具，用于比较自定义的多线程最近邻缩放与 OpenCV 的 `resize`
函数。测试结果通常显示，多线程实现具有更高的性能，特别是在处理大图像和高缩放因子的情况下。

示例输出：

```
输入图像尺寸: 400x300
图像通道数: 3

=== Test1_Upscale ===
输出尺寸: 800x600
自实现多线程最近邻缩放平均时间: 25 ms
OpenCV 最近邻缩放平均时间: 40 ms

=== Test2_Downscale ===
输出尺寸: 200x150
自实现多线程最近邻缩放平均时间: 20 ms
OpenCV 最近邻缩放平均时间: 35 ms

=== Test3_AspectRatioChange ===
输出尺寸: 300x450
自实现多线程最近邻缩放平均时间: 22 ms
OpenCV 最近邻缩放平均时间: 38 ms
  ```

7. 双线性插值

   7.1 运行可执行文件

   ./bilinear_interpolation

   7.2 行为

    - 从当前目录读取 `input_image.jpg`。
    - 将图像缩放到预定义尺寸（例如，800x600）。
    - 将缩放后的图像保存为 `resized_image_basic.jpg`。

   7.3 性能分析
   同时提供了另一个性能基准测试工具`Comparison_and_Analysis_with_ffmpeg.cpp`
   ，用于比较自定义的，OpenCV内置的和FFmpeg内置的最近邻缩放和双线性插值缩放。测试结果通常显示，多线程实现具有更高的性能，特别是在处理大图像和高缩放因子的情况下。

   示例输出：

    ```
   输入图像尺寸: 600x600
   图像通道数: 3
   
   === Test1_Upscale ===
   输出尺寸: 1200x1200
   自实现多线程最近邻缩放平均时间: 8 ms
   OpenCV 最近邻缩放平均时间: 0.5 ms
   FFmpeg 最近邻缩放平均时间: 5.7 ms
   自实现双线性插值缩放平均时间: 400 ms
   OpenCV 双线性插值缩放平均时间: 0.3 ms
   FFmpeg 双线性插值缩放平均时间: 5.7 ms
   
   图像质量比较 (PSNR，以OpenCV为基准):
   自实现最近邻 PSNR: 27.8962 dB
   FFmpeg PSNR: 361.202 dB
   自实现双线性插值 PSNR: 36.126 dB
   OpenCV 双线性插值 PSNR: 36.6253 dB
   FFmpeg 双线性插值 PSNR: 36.6261 dB
   
   === Test2_Downscale ===
   输出尺寸: 300x300
   自实现多线程最近邻缩放平均时间: 0.6 ms
   OpenCV 最近邻缩放平均时间: 0.1 ms
   FFmpeg 最近邻缩放平均时间: 0.6 ms
   自实现双线性插值缩放平均时间: 24.9 ms
   OpenCV 双线性插值缩放平均时间: 0.1 ms
   FFmpeg 双线性插值缩放平均时间: 0.9 ms
   
   图像质量比较 (PSNR，以OpenCV为基准):
   自实现最近邻 PSNR: 361.202 dB
   FFmpeg PSNR: 25.111 dB
   自实现双线性插值 PSNR: 30.6589 dB
   OpenCV 双线性插值 PSNR: 30.6918 dB
   FFmpeg 双线性插值 PSNR: 30.2388 dB
   
   === Test3_AspectRatioChange ===
   输出尺寸: 450x750
   自实现多线程最近邻缩放平均时间: 2 ms
   OpenCV 最近邻缩放平均时间: 0.1 ms
   FFmpeg 最近邻缩放平均时间: 1.5 ms
   自实现双线性插值缩放平均时间: 93.3 ms
   OpenCV 双线性插值缩放平均时间: 0.1 ms
   FFmpeg 双线性插值缩放平均时间: 1.5 ms
   
   图像质量比较 (PSNR，以OpenCV为基准):
   自实现最近邻 PSNR: 29.2712 dB
   FFmpeg PSNR: 29.2712 dB
   自实现双线性插值 PSNR: 30.9772 dB
   OpenCV 双线性插值 PSNR: 31.0103 dB
   FFmpeg 双线性插值 PSNR: 31.2208 dB
   ```
   
8. 进一步优化

算法增强：探索更高效的数据访问模式或使用 SIMD 指令来进一步加速缩放过程。
并行粒度调整：根据图像大小和硬件特性调整并行任务的粒度，以达到最佳并行效率。

- 内存管理：优化内存使用，减少缓存未命中，提高内存访问速度。

