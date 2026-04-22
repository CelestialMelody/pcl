/*
 * bench_gaussian.cpp
 *
 * 仅对 common/src/gaussian.cpp 中实现的成员函数做计时（公开 API）：
 *   - GaussianKernel::compute（两参数：生成核，供卷积使用）
 *   - convolveRows / convolveCols（PointCloud<float>）
 *   - 可分离两步：行卷积后再列卷积（语义上接近 smooth 的核心代价，但不经过 gaussian.h 中带调试输出的 convolve 内联）
 *
 * 修改 gaussian.cpp 后需重编安装 libpcl_common，再链接本 bench，结果才反映新实现。
 */

#include <pcl/common/gaussian.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

constexpr std::size_t kBenchmarkBannerWidth = 110;

static void
printBanner (char ch, std::size_t width = kBenchmarkBannerWidth)
{
  std::cout << std::string (width, ch) << '\n';
}

template <typename T>
static inline void
doNotOptimize (const T &value)
{
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r,m"(value) : "memory");
#else
  (void)value;
#endif
}

class Benchmarker
{
public:
  explicit Benchmarker (const std::string &name) : name_(name) {}

  void
  run (const std::function<void ()> &func, int iterations = 8, int warmup = 2)
  {
    for (int i = 0; i < warmup; ++i)
      func ();
    const auto start = std::chrono::high_resolution_clock::now ();
    for (int i = 0; i < iterations; ++i)
      func ();
    const auto end = std::chrono::high_resolution_clock::now ();
    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count ();
    const double avg_ms = total_ms / iterations;
    std::cout << std::left << std::setw (52) << name_ << ": " << std::fixed << std::setprecision (4)
              << avg_ms << " ms/iter\n";
  }

private:
  std::string name_;
};

static void
fillRandomImage (pcl::PointCloud<float> &img, std::uint32_t width, std::uint32_t height,
                  std::uint32_t seed = 42)
{
  img.width = width;
  img.height = height;
  img.resize (width * height);
  std::mt19937 rng (seed);
  std::uniform_real_distribution<float> u (0.0f, 1.0f);
  for (std::size_t i = 0; i < img.size (); ++i)
    img[i] = u (rng);
}

/** 对应 gaussian.cpp：compute(sigma, kernel) — 卷积前准备核，单独计时便于观察核生成成本占比 */
static void
runBenches (pcl::GaussianKernel &gk, const Eigen::VectorXf &kernel, float sigma,
            const pcl::PointCloud<float> &input, int iterations)
{
  pcl::PointCloud<float> tmp, out;
  Eigen::VectorXf mutable_kernel;

  {
    Benchmarker b ("compute(sigma, kernel) [gaussian.cpp]");
    b.run (
        [&] () {
          gk.compute (sigma, mutable_kernel);
          doNotOptimize (mutable_kernel[0]);
        },
        iterations);
  }

  {
    Benchmarker b ("convolveRows (PointCloud<float>) [gaussian.cpp]");
    b.run (
        [&] () {
          gk.convolveRows (input, kernel, out);
          doNotOptimize (out[0]);
        },
        iterations);
  }

  {
    Benchmarker b ("convolveCols (PointCloud<float>) [gaussian.cpp]");
    b.run (
        [&] () {
          gk.convolveCols (input, kernel, out);
          doNotOptimize (out[0]);
        },
        iterations);
  }

  {
    Benchmarker b ("convolveRows then convolveCols [gaussian.cpp x2]");
    b.run (
        [&] () {
          gk.convolveRows (input, kernel, tmp);
          gk.convolveCols (tmp, kernel, out);
          doNotOptimize (out[0]);
        },
        iterations);
  }
}

int
main (int argc, char **argv)
{
  /* 默认较小分辨率 + 较少迭代，便于板卡/QEMU 快速冒烟；完整基线可传参，例如：1920 1080 20 5 */
  std::uint32_t width = 960;
  std::uint32_t height = 540;
  int iterations = 10;
  float sigma = 5.0f;

  if (argc >= 2)
    width = static_cast<std::uint32_t> (std::atoi (argv[1]));
  if (argc >= 3)
    height = static_cast<std::uint32_t> (std::atoi (argv[2]));
  if (argc >= 4)
    iterations = std::atoi (argv[3]);
  if (argc >= 5)
    sigma = std::strtof (argv[4], nullptr);

  printBanner ('=');
  std::cout << " PCL gaussian.cpp Benchmark (GaussianKernel non-template members)\n";
  std::cout << " Image Size: " << width << " x " << height << "\n";
  std::cout << " Iterations: " << iterations << "\n";
  std::cout << " Dataset: random [0,1] float image; sigma=" << sigma << " for compute + convolve\n";
  std::cout << " Args: WIDTH HEIGHT ITERS SIGMA (omit for quick defaults; e.g. 1920 1080 20 5 for full HD)\n";
  std::cout << " Note: timings use symbols from libpcl_common (rebuild after editing gaussian.cpp)\n";
#if defined(__RVV10__)
  std::cout << " build: __RVV10__ defined\n";
#else
  std::cout << " build: __RVV10__ NOT defined\n";
#endif
  printBanner ('=');

  pcl::GaussianKernel gk;
  Eigen::VectorXf kernel;
  gk.compute (sigma, kernel);

  pcl::PointCloud<float> input;
  fillRandomImage (input, width, height, 42);

  runBenches (gk, kernel, sigma, input, iterations);

  printBanner ('=');
  return 0;
}
