/*
 * bench_centroid.cpp
 * PCL centroid.hpp 基准测试（聚焦高优先级候选）
 */

#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

/** 分隔线宽度；与 `analyze_bench_compare.py` 汇总表总宽约 100–110 列时视觉对齐（可按需改大）。 */
constexpr std::size_t kBenchmarkBannerWidth = 110;

static void
printBanner (char ch, std::size_t width = kBenchmarkBannerWidth)
{
  std::cout << std::string (width, ch) << '\n';
}

template <typename T>
static inline void doNotOptimize(const T& value)
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
  explicit Benchmarker(const std::string& name) : name_(name) {}

  void run(const std::function<void()>& func, int iterations = 20, int warmup = 3)
  {
    for (int i = 0; i < warmup; ++i) func();
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) func();
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double avg_ms = total_ms / iterations;
    std::cout << std::left << std::setw(52) << name_
              << ": " << std::fixed << std::setprecision(4) << avg_ms << " ms/iter\n";
  }

private:
  std::string name_;
};

static void fillRandomCloud(PointCloud& cloud, std::size_t n, std::uint32_t seed = 42)
{
  cloud.clear();
  cloud.resize(n);
  cloud.is_dense = true;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-10.0f, 10.0f);
  for (std::size_t i = 0; i < n; ++i)
  {
    cloud[i].x = u(rng);
    cloud[i].y = u(rng);
    cloud[i].z = u(rng);
  }
}

static void fillIndices(pcl::Indices& indices, std::size_t n)
{
  indices.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    indices[i] = static_cast<int>(i);
}

static void runBenches(const PointCloud& cloud, const pcl::Indices& indices, int iterations)
{
  {
    Benchmarker b("compute3DCentroid (cloud)");
    b.run([&]() {
      Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
      const auto count = pcl::compute3DCentroid(cloud, centroid);
      doNotOptimize(count);
      doNotOptimize(centroid);
    }, iterations);
  }

  {
    Benchmarker b("compute3DCentroid (indices)");
    b.run([&]() {
      Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
      const auto count = pcl::compute3DCentroid(cloud, indices, centroid);
      doNotOptimize(count);
      doNotOptimize(centroid);
    }, iterations);
  }

  {
    Benchmarker b("computeMeanAndCovarianceMatrix (cloud)");
    b.run([&]() {
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
      const auto count = pcl::computeMeanAndCovarianceMatrix(cloud, cov, centroid);
      doNotOptimize(count);
      doNotOptimize(cov);
      doNotOptimize(centroid);
    }, iterations);
  }

  {
    Benchmarker b("computeMeanAndCovarianceMatrix (indices)");
    b.run([&]() {
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
      const auto count = pcl::computeMeanAndCovarianceMatrix(cloud, indices, cov, centroid);
      doNotOptimize(count);
      doNotOptimize(cov);
      doNotOptimize(centroid);
    }, iterations);
  }

  {
    Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
    pcl::compute3DCentroid(cloud, centroid);
    Benchmarker b("computeCovarianceMatrix (cloud, given centroid)");
    b.run([&]() {
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      const auto count = pcl::computeCovarianceMatrix(cloud, centroid, cov);
      doNotOptimize(count);
      doNotOptimize(cov);
    }, iterations);
  }

  {
    Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
    pcl::compute3DCentroid(cloud, centroid);
    Benchmarker b("computeCovarianceMatrix (indices, given centroid)");
    b.run([&]() {
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      const auto count =
          pcl::computeCovarianceMatrix(cloud, indices, centroid, cov);
      doNotOptimize(count);
      doNotOptimize(cov);
    }, iterations);
  }

  {
    Benchmarker b("computeCovarianceMatrix (cloud, about origin)");
    b.run([&]() {
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      const auto count = pcl::computeCovarianceMatrix(cloud, cov);
      doNotOptimize(count);
      doNotOptimize(cov);
    }, iterations);
  }

  {
    Benchmarker b("computeCovarianceMatrix (indices, about origin)");
    b.run([&]() {
      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      const auto count = pcl::computeCovarianceMatrix(cloud, indices, cov);
      doNotOptimize(count);
      doNotOptimize(cov);
    }, iterations);
  }

  // demeanPointCloud：稠密 + n>>16 时可走 RVV（整云 in-place 写 cloud_out；indices 为 gather+写回；Eigen 为 SoA 写前三行）
  {
    Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
    pcl::compute3DCentroid(cloud, centroid);
    PointCloud cloud_out;
    Benchmarker b("demeanPointCloud (cloud -> PointCloud)");
    b.run([&]() {
      pcl::demeanPointCloud(cloud, centroid, cloud_out);
      doNotOptimize(cloud_out[0].x);
    }, iterations);
  }

  {
    Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
    pcl::compute3DCentroid(cloud, centroid);
    PointCloud cloud_out;
    Benchmarker b("demeanPointCloud (indices -> PointCloud)");
    b.run([&]() {
      pcl::demeanPointCloud(cloud, indices, centroid, cloud_out);
      doNotOptimize(cloud_out[0].x);
    }, iterations);
  }

  {
    Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
    pcl::compute3DCentroid(cloud, centroid);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_out;
    Benchmarker b("demeanPointCloud (cloud -> Eigen 4xN)");
    b.run([&]() {
      pcl::demeanPointCloud(cloud, centroid, mat_out);
      doNotOptimize(mat_out(0, 0));
    }, iterations);
  }
}

int main(int argc, char** argv)
{
  std::size_t points = 1'000'000;
  int iterations = 20;

  if (argc >= 2) points = static_cast<std::size_t>(std::atoi(argv[1]));
  if (argc >= 3) iterations = std::atoi(argv[2]);

  printBanner ('=');
  std::cout << " PCL centroid.hpp Benchmark\n";
  std::cout << " Cloud size: " << points << "\n";
  std::cout << " Iterations: " << iterations << "\n";
  std::cout << " Dataset: points=" << points << " seed=42\n";
#if defined(__RVV10__)
  std::cout << " build: __RVV10__ defined\n";
#else
  std::cout << " build: __RVV10__ NOT defined\n";
#endif
  printBanner ('=');

  PointCloud cloud;
  fillRandomCloud(cloud, points, 42);

  pcl::Indices indices;
  fillIndices(indices, cloud.size());

  runBenches(cloud, indices, iterations);

  printBanner ('=');
  return 0;
}

