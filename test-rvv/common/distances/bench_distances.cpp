/**
 * bench_distances.cpp
 * PCL common/distances.h 基准测试
 *
 * 覆盖：
 *   - squaredEuclideanDistance / euclideanDistance
 *   - sqrPointToLineDistance
 *   - getMaxSegment（O(n^2)，使用更小规模）
 */

#include <pcl/common/distances.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

// ============================================================================
// 防止基准计算被优化掉
// ============================================================================
template <typename T>
static inline void doNotOptimize(const T& value)
{
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r,m"(value) : "memory");
#else
  (void)value;
#endif
}

// ============================================================================
// 计时
// ============================================================================
class Benchmarker {
public:
  explicit Benchmarker(const std::string& name) : name_(name) {}

  void run(const std::function<void()>& func, int iterations = 20, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) func();
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) func();
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double avg_ms = total_ms / iterations;
    std::cout << std::left << std::setw(42) << name_
              << ": " << std::fixed << std::setprecision(4) << avg_ms << " ms/iter\n";
  }

private:
  std::string name_;
};

// ============================================================================
// 数据生成
// ============================================================================
static void fillRandomCloud(PointCloud& cloud, std::size_t n, std::uint32_t seed = 42) {
  cloud.clear();
  cloud.resize(n);
  cloud.is_dense = true;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-10.0f, 10.0f);
  for (std::size_t i = 0; i < n; ++i) {
    cloud[i].x = u(rng);
    cloud[i].y = u(rng);
    cloud[i].z = u(rng);
  }
}

static void fillIndices(pcl::Indices& indices, std::size_t n) {
  indices.resize(n);
  for (std::size_t i = 0; i < n; ++i) indices[i] = static_cast<int>(i);
}

// ============================================================================
// Bench functions
// ============================================================================
static void runLinearBenches(const PointCloud& cloud_linear, int iterations)
{
  // ---- squaredEuclideanDistance / euclideanDistance（线性）----
  {
    Benchmarker b("squaredEuclideanDistance (PointXYZ, xN)");
    b.run([&]() {
      double acc = 0.0;
      for (std::size_t i = 1; i < cloud_linear.size(); ++i) {
        acc += pcl::squaredEuclideanDistance(cloud_linear[i - 1], cloud_linear[i]);
      }
      doNotOptimize(acc);
    }, iterations);
  }

  {
    Benchmarker b("euclideanDistance (PointXYZ, xN)");
    b.run([&]() {
      double acc = 0.0;
      for (std::size_t i = 1; i < cloud_linear.size(); ++i) {
        acc += pcl::euclideanDistance(cloud_linear[i - 1], cloud_linear[i]);
      }
      doNotOptimize(acc);
    }, iterations);
  }

  // ---- sqrPointToLineDistance（线性）----
  {
    const Eigen::Vector4f line_pt(0.0f, 0.0f, 0.0f, 0.0f);
    const Eigen::Vector4f line_dir(1.0f, 1.0f, 0.0f, 0.0f);
    const double sqr_length = static_cast<double>(line_dir.squaredNorm());

    Benchmarker b("sqrPointToLineDistance (xN)");
    b.run([&]() {
      double acc = 0.0;
      for (std::size_t i = 0; i < cloud_linear.size(); ++i) {
        const Eigen::Vector4f pt(cloud_linear[i].x, cloud_linear[i].y, cloud_linear[i].z, 0.0f);
        acc += pcl::sqrPointToLineDistance(pt, line_pt, line_dir, sqr_length);
      }
      doNotOptimize(acc);
    }, iterations);
  }
}

static void runMaxSegmentBenches(std::size_t maxseg_points, int iterations)
{
  // ---- getMaxSegment（O(n^2)）----
  {
    PointCloud cloud;
    fillRandomCloud(cloud, maxseg_points, 7);
    PointT pmin, pmax;
    Benchmarker b("getMaxSegment (cloud, O(n^2))");
    b.run([&]() {
      const double len = pcl::getMaxSegment(cloud, pmin, pmax);
      const double acc = len + static_cast<double>(pmin.x) + static_cast<double>(pmax.x);
      doNotOptimize(acc);
    }, std::max(1, iterations / 4));
  }

  // ---- getMaxSegment（indices 版，O(n^2)）----
  {
    PointCloud cloud;
    fillRandomCloud(cloud, maxseg_points, 11);
    pcl::Indices indices;
    fillIndices(indices, cloud.size());
    PointT pmin, pmax;
    Benchmarker b("getMaxSegment (indices, O(n^2))");
    b.run([&]() {
      const double len = pcl::getMaxSegment(cloud, indices, pmin, pmax);
      const double acc = len + static_cast<double>(pmin.y) + static_cast<double>(pmax.y);
      doNotOptimize(acc);
    }, std::max(1, iterations / 4));
  }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
  std::size_t linear_points = 1000000;  // 线性循环：用于测吞吐
  std::size_t maxseg_points = 2500;     // O(n^2)：用于 getMaxSegment
  int iterations = 20;

  if (argc >= 2) linear_points = static_cast<std::size_t>(std::atoi(argv[1]));
  if (argc >= 3) maxseg_points = static_cast<std::size_t>(std::atoi(argv[2]));
  if (argc >= 4) iterations = std::atoi(argv[3]);

  std::cout << "============================================================\n";
  std::cout << " PCL distances.h Benchmark\n";
  std::cout << " linear_points: " << linear_points << "  maxseg_points: " << maxseg_points << "\n";
  std::cout << " iterations: " << iterations << " iter: " << iterations << "\n";
  std::cout << " Dataset: linear_points=" << linear_points
            << " maxseg_points=" << maxseg_points
            << " iterations=" << iterations
            << " maxseg_seeds=(7,11)\n";
#if defined(__RVV10__)
  std::cout << " build: __RVV10__ defined\n";
#else
  std::cout << " build: __RVV10__ NOT defined\n";
#endif
  std::cout << "============================================================\n";

  PointCloud cloud_linear;
  fillRandomCloud(cloud_linear, linear_points);

  // runLinearBenches(cloud_linear, iterations); // 保留用于静态分析/吞吐对照，默认不执行
  runMaxSegmentBenches(maxseg_points, iterations);

  std::cout << "============================================================\n";
  return 0;
}

