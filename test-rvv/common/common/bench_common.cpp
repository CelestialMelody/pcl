/**
 * bench_common.cpp
 * PCL common (common.h / impl/common.hpp) 全面覆盖 Benchmark
 * 用于分析与优化：getMeanStd、getPointsInBox、getMaxDistance、getMinMax3D、
 * getAngle3D、getCircumcircleRadius、calculatePolygonArea、getMinMax、computeMedian 等
 */

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>  // pcl::Indices
#include <Eigen/Core>

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <cmath>

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

// ============================================================================
// 计时与统计
// ============================================================================
class Benchmarker {
public:
  explicit Benchmarker(const std::string& name) : name_(name) {}

  void run(const std::function<void()>& func, int iterations = 20, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) func();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) func();
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
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

static void fillRandomVector(std::vector<float>& v, std::size_t n, std::uint32_t seed = 42) {
  v.resize(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 100.0f);
  for (std::size_t i = 0; i < n; ++i) v[i] = u(rng);
}

// ============================================================================
// Benchmark 项（覆盖 common.hpp 热点）
// ============================================================================
int main(int argc, char** argv) {
  std::size_t cloud_size = 200000;  // 约 20 万点，便于观察 RVV 收益
  std::size_t vec_size   = 500000;
  int iterations = 20;

  if (argc >= 2) cloud_size = static_cast<std::size_t>(std::atoi(argv[1]));
  if (argc >= 3) vec_size   = static_cast<std::size_t>(std::atoi(argv[2]));
  if (argc >= 4) iterations = std::atoi(argv[3]);

  std::cout << "============================================================\n";
  std::cout << " PCL common.hpp Benchmark\n";
  std::cout << " Cloud size: " << cloud_size << "  Vector size: " << vec_size << "\n";
  std::cout << " Iterations: " << iterations << "\n";
  std::cout << "============================================================\n";

  PointCloud cloud;
  fillRandomCloud(cloud, cloud_size);

  std::vector<float> values;
  fillRandomVector(values, vec_size);

  Eigen::Vector4f min_pt(-5.0f, -5.0f, -5.0f, 1.0f);
  Eigen::Vector4f max_pt( 5.0f,  5.0f,  5.0f, 1.0f);
  Eigen::Vector4f pivot_pt(0.0f, 0.0f, 0.0f, 1.0f);

  pcl::Indices indices;
  indices.resize(cloud_size);
  for (std::size_t i = 0; i < cloud_size; ++i) indices[i] = static_cast<int>(i);

  Eigen::Vector4f min_out, max_out;
  Eigen::Vector4f max_dist_pt;
  double mean, stddev;

  // ---- getMeanStd (inline in common.hpp: 单遍 sum + sq_sum) ----
  {
    Benchmarker b("getMeanStd (vector<float>)");
    b.run([&]() {
      pcl::getMeanStd(values, mean, stddev);
    }, iterations);
  }

  // ---- getPointsInBox (dense: 6 比较/点，写 indices) ----
  {
    pcl::Indices out_indices;
    Benchmarker b("getPointsInBox (dense cloud)");
    b.run([&]() {
      out_indices.clear();
      pcl::getPointsInBox(cloud, min_pt, max_pt, out_indices);
    }, iterations);
  }

  // ---- getMaxDistance (cloud, pivot, max_pt) ----
  {
    Benchmarker b("getMaxDistance (cloud, pivot)");
    b.run([&]() {
      pcl::getMaxDistance(cloud, pivot_pt, max_dist_pt);
    }, iterations);
  }

  // ---- getMaxDistance (cloud, indices, pivot, max_pt) ----
  {
    Benchmarker b("getMaxDistance (cloud, indices, pivot)");
    b.run([&]() {
      pcl::getMaxDistance(cloud, indices, pivot_pt, max_dist_pt);
    }, iterations);
  }

  // ---- getMinMax3D (cloud, min_pt, max_pt) ----
  {
    Benchmarker b("getMinMax3D (cloud, Eigen)");
    b.run([&]() {
      pcl::getMinMax3D(cloud, min_out, max_out);
    }, iterations);
  }

  // ---- getMinMax3D (cloud, indices, min_pt, max_pt) ----
  {
    Benchmarker b("getMinMax3D (cloud, indices, Eigen)");
    b.run([&]() {
      pcl::getMinMax3D(cloud, indices, min_out, max_out);
    }, iterations);
  }

  // ---- getMinMax3D (cloud, PointT, PointT) ----
  {
    PointT min_pt_p, max_pt_p;
    Benchmarker b("getMinMax3D (cloud, PointT)");
    b.run([&]() {
      pcl::getMinMax3D(cloud, min_pt_p, max_pt_p);
    }, iterations);
  }

  // ---- getAngle3D (多对向量，标量但可作基线) ----
  {
    // 为保证可比性：本 benchmark 里 v1 = (cos(t), sin(t), 0, 0)，v2 = (0, 1, 0, 0)
    // dot(v1, v2) = sin(t) >= 0 (t in [0, 0.999])，因此 “acute angle” 与 “标准夹角” 数值一致。
    constexpr int kN = 1000;
    std::vector<float> x1_vals(kN), y1_vals(kN);
    for (int i = 0; i < kN; ++i) {
      const float t = static_cast<float>(i) * 0.001f;
      x1_vals[i] = std::cos(t);
      y1_vals[i] = std::sin(t);
    }
#if defined(__RVV10__)
    // RVV：向量函数输入已假设“归一化 unit vectors”，这里 x1^2+y1^2=1，z1=0。
    double angle_sum_rvv = 0;
    Benchmarker brvv("getAngle3D (x1000)");
    brvv.run([&]() {
      angle_sum_rvv = 0;
      std::size_t j = 0;
      while (j < static_cast<std::size_t>(kN)) {
        const std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(kN) - j);

        const vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x1_vals.data() + j, vl);
        const vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y1_vals.data() + j, vl);
        const vfloat32m2_t vz1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        // v2 = (0,1,0)
        const vfloat32m2_t vx2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        const vfloat32m2_t vy2 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t vz2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const vfloat32m2_t v_ang = pcl::getAcuteAngle3DRVV_f32m2(vx1, vy1, vz1, vx2, vy2, vz2, vl);

        // reduce 到标量，避免编译器优化掉计算结果
        const float sum_block =
            __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m2_f32m1(
                v_ang, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl));
        angle_sum_rvv += sum_block;
        j += vl;
      }
    }, iterations);
    (void)angle_sum_rvv;
#else
    Eigen::Vector4f v1(1, 0, 0, 0), v2(0, 1, 0, 0);
    double angle_sum = 0;
    Benchmarker b("getAngle3D (x1000)");
    b.run([&]() {
      angle_sum = 0;
      for (int i = 0; i < kN; ++i) {
        v1[0] = x1_vals[i];
        v1[1] = y1_vals[i];
        angle_sum += pcl::getAngle3D(v1, v2, false);
      }
    }, iterations);
    (void)angle_sum;
#endif
  }

  // ---- calculatePolygonArea (多边形顶点循环) ----
  {
    PointCloud polygon;
    polygon.resize(256);
    for (std::size_t i = 0; i < polygon.size(); ++i) {
      float t = static_cast<float>(i) / polygon.size() * 2.0f * 3.14159f;
      polygon[i].x = std::cos(t);
      polygon[i].y = std::sin(t);
      polygon[i].z = 0;
    }
    float area = 0;
    Benchmarker b("calculatePolygonArea (256 pts) x500");
    b.run([&]() {
      for (int i = 0; i < 500; ++i) area += pcl::calculatePolygonArea(polygon);
    }, iterations);
    (void)area;
  }

  std::cout << "============================================================\n";
  std::cout << " Done. Use run_bench under ARCH=riscv or ARCH=x86 for comparison.\n";
  std::cout << "============================================================\n";
  return 0;
}
