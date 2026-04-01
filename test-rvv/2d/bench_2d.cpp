/**
 * bench_2d.cpp
 * PCL 2D Module Benchmark for RISC-V SIMD Optimization
 * Covers: Kernel generation, Convolution, Edge, Morphology
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/2d/convolution.h>
#include <pcl/2d/edge.h>
#include <pcl/2d/morphology.h>
#include <pcl/2d/kernel.h>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <functional>
#include <cstddef>
#include <algorithm>

// 使用 PointXYZI 因为 2D 模块主要操作 intensity
using PointT = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointT>;

// 用于 Edge 模块输出的类型，包含 magnitude, direction 等字段
using PointEdgeT = pcl::PointXYZIEdge;
using PointCloudEdge = pcl::PointCloud<PointEdgeT>;

// 防止 -O3 下计时循环内的计算被当成无副作用而整体删除：经 volatile 累加后编译器必须保留对结果的依赖。
static volatile float g_sink_f32 = 0.0f;

static inline void
consumeFloat(float v)
{
  g_sink_f32 += v * 1.0000001f;
}

// ============================================================================
// 辅助类：计时器与统计
// ============================================================================
class Benchmarker {
public:
  Benchmarker(std::string name) : name_(name) {}

  void run(std::function<void()> func, int iterations = 10, int warmup = 2) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
      func();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    const double total_ns =
        std::chrono::duration<double, std::nano>(end - start).count();
    const double avg_ns = total_ns / iterations;

    const bool use_us = avg_ns < 1e6; // < 1ms
    const double value = use_us ? (avg_ns / 1e3) : (avg_ns / 1e6);
    const char* unit = use_us ? " us / iter" : " ms / iter";

    std::cout << std::left << std::setw(40) << name_
              << ": " << std::fixed << std::setprecision(3)
              << value << unit << std::endl;
  }

private:
  std::string name_;
};

// ============================================================================
// 数据生成
// ============================================================================
void generateRandomImage(PointCloud::Ptr& cloud, int width, int height) {
  cloud->width = width;
  cloud->height = height;
  cloud->resize(width * height);
  cloud->is_dense = false;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 255.0f);

  for (auto& p : cloud->points) {
    p.intensity = dist(rng);
    p.x = p.y = p.z = 0.0f; // 2D 模块不使用 xyz，但在某些函数内部可能会被访问
  }
}

// ============================================================================
// Main Benchmark
// ============================================================================
int main(int argc, char** argv) {
  int width = 640;
  int height = 480;
  int iterations = 20;

  if (argc >= 3) {
    width = std::atoi(argv[1]);
    height = std::atoi(argv[2]);
  }

  std::cout << "========================================================" << std::endl;
  std::cout << " PCL 2D Module Benchmark" << std::endl;
  std::cout << " Image Size: " << width << " x " << height << std::endl;
  std::cout << " Iterations: " << iterations << std::endl;
  std::cout << "========================================================" << std::endl;

  // 准备数据
  PointCloud::Ptr input(new PointCloud);
  PointCloud::Ptr output(new PointCloud); // 通用输出 (Convolution, Morphology)
  PointCloudEdge::Ptr output_edge(new PointCloudEdge); // Edge 专用输出
  PointCloud::Ptr kernel_cloud(new PointCloud);
  generateRandomImage(input, width, height);

  // 准备第二个输入用于集合操作
  PointCloud::Ptr input2(new PointCloud);
  generateRandomImage(input2, width, height);
  // 二值化用于集合操作测试
  for(auto& p : input->points) p.intensity = (p.intensity > 128 ? 1.0f : 0.0f);
  for(auto& p : input2->points) p.intensity = (p.intensity > 128 ? 1.0f : 0.0f);

  // --------------------------------------------------------------------------
  // 0. Kernel Generation Benchmark (gaussianKernel / loGKernel)
  // --------------------------------------------------------------------------
  {
    pcl::kernel<PointT> k;

    struct KernelCase {
      const char* name;
      int size;
      float sigma;
    };

    const KernelCase cases[] = {
      {"Kernel gaussianKernel (3x3, sigma=1.0)", 3, 1.0f},
      {"Kernel gaussianKernel (5x5, sigma=1.0)", 5, 1.0f},
      {"Kernel gaussianKernel (11x11, sigma=2.0)", 11, 2.0f},
      {"Kernel loGKernel (3x3, sigma=1.0)", 3, 1.0f},
      {"Kernel loGKernel (5x5, sigma=1.0)", 5, 1.0f},
      {"Kernel loGKernel (11x11, sigma=2.0)", 11, 2.0f},
    };

    for (const auto& c : cases) {
      k.setKernelSize(c.size);
      k.setKernelSigma(c.sigma);

      Benchmarker(c.name).run([&]() {
        if (std::string(c.name).find("gaussianKernel") != std::string::npos) {
          k.gaussianKernel(*kernel_cloud);
        }
        else {
          k.loGKernel(*kernel_cloud);
        }
        // 读回写入 cloud 的一个标量，避免 kernel 生成被 DCE；与 consumeFloat 配合固定计时对象。
        if (!kernel_cloud->empty())
          consumeFloat((*kernel_cloud)[0].intensity);
      }, iterations * 20);
    }
  }

  // --------------------------------------------------------------------------
  // 1. Convolution Benchmark
  // --------------------------------------------------------------------------
  {
    pcl::Convolution<PointT> conv;
    conv.setInputCloud(input);

    // 生成一个 5x5 高斯核
    pcl::kernel<PointT> k;
    k.setKernelType(pcl::kernel<PointT>::GAUSSIAN);
    k.setKernelSize(5);
    k.setKernelSigma(1.0f);
    k.fetchKernel(*kernel_cloud);
    conv.setKernel(*kernel_cloud);

    Benchmarker("Convolution (5x5 Gaussian)").run([&](){
      conv.filter(*output);
    }, iterations);
  }

  // --------------------------------------------------------------------------
  // 1.1 Convolution-only Benchmark (Kernel 生成与计时解耦)
  // --------------------------------------------------------------------------
  {
    // 11x11 卷积计算量显著更大，这里减少迭代次数，避免单次 benchmark 过慢。
    const int conv11_iterations = std::max(1, iterations / 4);

    pcl::Convolution<PointT> conv;
    conv.setInputCloud(input);

    pcl::kernel<PointT> k;
    k.setKernelType(pcl::kernel<PointT>::GAUSSIAN);
    k.setKernelSize(11);
    k.setKernelSigma(2.0f);

    // 只在计时外生成一次 kernel，测量重点是卷积本身。
    k.fetchKernel(*kernel_cloud);
    conv.setKernel(*kernel_cloud);

    Benchmarker("Convolution (11x11 Gaussian)").run([&](){
      conv.filter(*output);
    }, conv11_iterations);
  }

  // --------------------------------------------------------------------------
  // 1.5 End-to-end Benchmark (Kernel generation + setKernel + Convolution)
  // --------------------------------------------------------------------------
  {
    // Kernel generation on RVV shows the largest benefit on 11x11.
    // Here we regenerate kernel each iteration to quantify end-to-end impact.
    const int end2end_iterations_11x11 = std::max(1, iterations / 4);
    const int end2end_iterations_5x5 = iterations; // 5x5 卷积更快，可用满迭代

    pcl::Convolution<PointT> conv;
    conv.setInputCloud(input);
    pcl::kernel<PointT> k_end;

    auto runEnd2EndGaussian = [&](int kernel_size, float sigma) {
      k_end.setKernelSize(kernel_size);
      k_end.setKernelSigma(sigma);
      k_end.gaussianKernel(*kernel_cloud);

      conv.setKernel(*kernel_cloud);
      conv.filter(*output);
      if (!output->empty())
        consumeFloat((*output)[0].intensity);
    };

    Benchmarker("Conv+Kernel Gaussian (5x5, sigma=1.0)").run(
        [&]() { runEnd2EndGaussian(5, 1.0f); }, end2end_iterations_5x5);

    Benchmarker("Conv+Kernel Gaussian (11x11, sigma=2.0)").run(
        [&]() { runEnd2EndGaussian(11, 2.0f); }, end2end_iterations_11x11);

    auto runEnd2EndLoG = [&](int kernel_size, float sigma) {
      k_end.setKernelSize(kernel_size);
      k_end.setKernelSigma(sigma);
      k_end.loGKernel(*kernel_cloud);

      conv.setKernel(*kernel_cloud);
      conv.filter(*output);
      if (!output->empty())
        consumeFloat((*output)[0].intensity);
    };

    Benchmarker("Conv+Kernel LoG (11x11, sigma=2.0)").run(
        [&]() { runEnd2EndLoG(11, 2.0f); }, end2end_iterations_11x11);
  }

  // --------------------------------------------------------------------------
  // 2. Edge Detection Benchmark
  // --------------------------------------------------------------------------
  {
    // 模板参数修改：输入 PointT (PointXYZI)，输出 PointEdgeT (PointXYZIEdge)
    pcl::Edge<PointT, PointEdgeT> edge;
    edge.setInputCloud(input);

    // 注意：output_edge 必须预先设置大小，还是由算法resize？
    // 通常 PCL 算法会负责 resize，但最好确保指针有效。

    Benchmarker("Edge Sobel (Magnitude)").run([&](){
      edge.detectEdgeSobel(*output_edge); // 使用 PointXYZIEdge 输出
    }, iterations);

    Benchmarker("Edge Canny (Full Pipeline)").run([&](){
      edge.detectEdgeCanny(*output_edge); // 使用 PointXYZIEdge 输出
    }, iterations);

    // ------------------------------------------------------------------------
    // 2.1 discretizeAngles：Std vs RVV
    // 每次迭代从同一 seed 拷贝，避免原地修改导致下一次计时失真。
    // ------------------------------------------------------------------------
    {
      PointCloudEdge seed;
      seed.width = width;
      seed.height = height;
      seed.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
      std::mt19937 rng_dir(123);
      std::uniform_real_distribution<float> dist_rad(-3.14159265f, 3.14159265f);
      for (auto& p : seed.points)
        p.direction = dist_rad(rng_dir);

#if defined(__RVV10__)
      Benchmarker("Edge discretizeAngles").run(
          [&]() {
            PointCloudEdge tmp = seed;
            pcl::discretizeAnglesRVV(tmp, height, width);
            consumeFloat(tmp(0, 0).direction);
          },
          iterations);
#else
      Benchmarker("Edge discretizeAngles").run(
          [&]() {
            PointCloudEdge tmp = seed;
            pcl::discretizeAnglesStd(tmp, height, width);
            consumeFloat(tmp(0, 0).direction);
          },
          iterations);
#endif
    }
  }

  // --------------------------------------------------------------------------
  // 3. Morphology Benchmark
  // --------------------------------------------------------------------------
  {
    pcl::Morphology<PointT> morph;
    morph.setInputCloud(input);

    // 结构元素 3x3
    pcl::PointCloud<PointT> struct_elem;
    morph.structuringElementRectangle(struct_elem, 3, 3);
    morph.setStructuringElement(struct_elem.makeShared());

    // Erosion Gray
    Benchmarker("Morphology Erosion (Gray 3x3)").run([&](){
      morph.erosionGray(*output);
    }, iterations);

    // Dilation Gray
    Benchmarker("Morphology Dilation (Gray 3x3)").run([&](){
      morph.dilationGray(*output);
    }, iterations);

    // Opening Gray (Erosion + Dilation)
    Benchmarker("Morphology Opening (Gray 3x3)").run([&](){
      morph.openingGray(*output);
    }, iterations);

    // Closing Gray (Dilation + Erosion)
    Benchmarker("Morphology Closing (Gray 3x3)").run([&](){
      morph.closingGray(*output);
    }, iterations);

    // Binary morphology operations need binary input (0 or 1)
    PointCloud::Ptr input_binary(new PointCloud);
    input_binary->width = width;
    input_binary->height = height;
    input_binary->resize(width * height);
    std::mt19937 rng_binary(42);
    std::uniform_int_distribution<int> dist_binary(0, 1);
    for (std::size_t i = 0; i < input_binary->points.size(); ++i) {
      input_binary->points[i].intensity = static_cast<float>(dist_binary(rng_binary));
      input_binary->points[i].x = input_binary->points[i].y = input_binary->points[i].z = 0.0f;
    }
    morph.setInputCloud(input_binary);

    // Erosion Binary
    Benchmarker("Morphology Erosion (Binary 3x3)").run([&](){
      morph.erosionBinary(*output);
    }, iterations);

    // Dilation Binary
    Benchmarker("Morphology Dilation (Binary 3x3)").run([&](){
      morph.dilationBinary(*output);
    }, iterations);

    // Opening Binary (Erosion + Dilation)
    Benchmarker("Morphology Opening (Binary 3x3)").run([&](){
      morph.openingBinary(*output);
    }, iterations);

    // Closing Binary (Dilation + Erosion)，保证在同一输入上测
    morph.setInputCloud(input_binary);
    Benchmarker("Morphology Closing (Binary 3x3)").run([&](){
      morph.closingBinary(*output);
    }, iterations);
  }

  // --------------------------------------------------------------------------
  // 4. Morphology Set Operations Benchmark (Bandwidth Bound)
  // --------------------------------------------------------------------------
  {
    pcl::Morphology<PointT> morph;
    // Set Operations 不需要 setInputCloud，直接传参数

    Benchmarker("Set Operation: Union (A | B)").run([&](){
      morph.unionBinary(*output, *input, *input2);
    }, iterations * 5); // 运算很快，增加迭代次数

    Benchmarker("Set Operation: Intersection (A & B)").run([&](){
      morph.intersectionBinary(*output, *input, *input2);
    }, iterations * 5);

    Benchmarker("Set Operation: Subtraction (A - B)").run([&](){
      morph.subtractionBinary(*output, *input, *input2);
    }, iterations * 5);
  }

  std::cout << "========================================================" << std::endl;
  return 0;
}
