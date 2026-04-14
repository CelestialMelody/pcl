/**
 * Compare two RVV load strategies in getMaxSegmentRVV inner loop:
 *   - Mode A: 3x strided load (vlse32) for x/y/z
 *   - Mode B: 1x strided segment load (vlsseg3e32) for x/y/z
 *
 * This is a micro-benchmark to validate whether "fewer load instructions"
 * is beneficial and stable on a given board/compiler.
 */
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#if defined(__RVV10__)
#include <riscv_vector.h>
#endif

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

static volatile float g_sink_f32 = 0.0f;
static inline void consume(float v) { g_sink_f32 += v * 1.0000001f; }

static void fillRandomCloud(PointCloud& cloud, std::size_t n, std::uint32_t seed = 7)
{
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

#if defined(__RVV10__)
struct MaxSegResult {
  float max_dist2;
  std::size_t i_min;
  std::size_t i_max;
};

static inline MaxSegResult getMaxSegmentRVV_vlse(const PointCloud& cloud)
{
  const std::size_t n = cloud.size();
  float max_dist = -std::numeric_limits<float>::infinity();
  const auto token = std::numeric_limits<std::size_t>::max();
  std::size_t i_min = token, i_max = token;
  const ptrdiff_t stride = static_cast<ptrdiff_t>(sizeof(PointT));

  for (std::size_t i = 0; i < n; ++i) {
    const float xi = cloud[i].x;
    const float yi = cloud[i].y;
    const float zi = cloud[i].z;

    std::size_t j = i;
    while (j < n) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - j);
      const float* xptr = &cloud[j].x;
      const float* yptr = &cloud[j].y;
      const float* zptr = &cloud[j].z;

      vfloat32m2_t vx = __riscv_vlse32_v_f32m2(xptr, stride, vl);
      vfloat32m2_t vy = __riscv_vlse32_v_f32m2(yptr, stride, vl);
      vfloat32m2_t vz = __riscv_vlse32_v_f32m2(zptr, stride, vl);

      vx = __riscv_vfsub_vf_f32m2(vx, xi, vl);
      vy = __riscv_vfsub_vf_f32m2(vy, yi, vl);
      vz = __riscv_vfsub_vf_f32m2(vz, zi, vl);

      vfloat32m2_t vdist2 = __riscv_vfmul_vv_f32m2(vx, vx, vl);
      vdist2 = __riscv_vfmacc_vv_f32m2(vdist2, vy, vy, vl);
      vdist2 = __riscv_vfmacc_vv_f32m2(vdist2, vz, vz, vl);

      const vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1(max_dist, 1);
      const vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m2_f32m1(vdist2, vinit, vl);
      const float vmax = __riscv_vfmv_f_s_f32m1_f32(vmax1);

      if (vmax > max_dist) {
        const vbool16_t m = __riscv_vmfeq_vf_f32m2_b16(vdist2, vmax, vl);
        const long lane = __riscv_vfirst_m_b16(m, vl);
        if (lane >= 0) {
          max_dist = vmax;
          i_min = i;
          i_max = j + static_cast<std::size_t>(lane);
        }
      }

      j += vl;
    }
  }

  if (i_min == token || i_max == token)
    return {std::numeric_limits<float>::quiet_NaN(), token, token};

  return {max_dist, i_min, i_max};
}

static inline MaxSegResult getMaxSegmentRVV_vlsseg3(const PointCloud& cloud)
{
  const std::size_t n = cloud.size();
  float max_dist = -std::numeric_limits<float>::infinity();
  const auto token = std::numeric_limits<std::size_t>::max();
  std::size_t i_min = token, i_max = token;
  const ptrdiff_t stride = static_cast<ptrdiff_t>(sizeof(PointT));

  for (std::size_t i = 0; i < n; ++i) {
    const float xi = cloud[i].x;
    const float yi = cloud[i].y;
    const float zi = cloud[i].z;

    std::size_t j = i;
    while (j < n) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - j);
      const float* base = &cloud[j].x;

      // Load (x,y,z) with one strided segment-load.
      const vfloat32m2x3_t v_xyz = __riscv_vlsseg3e32_v_f32m2x3(base, stride, vl);
      vfloat32m2_t vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
      vfloat32m2_t vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
      vfloat32m2_t vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

      vx = __riscv_vfsub_vf_f32m2(vx, xi, vl);
      vy = __riscv_vfsub_vf_f32m2(vy, yi, vl);
      vz = __riscv_vfsub_vf_f32m2(vz, zi, vl);

      vfloat32m2_t vdist2 = __riscv_vfmul_vv_f32m2(vx, vx, vl);
      vdist2 = __riscv_vfmacc_vv_f32m2(vdist2, vy, vy, vl);
      vdist2 = __riscv_vfmacc_vv_f32m2(vdist2, vz, vz, vl);

      const vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1(max_dist, 1);
      const vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m2_f32m1(vdist2, vinit, vl);
      const float vmax = __riscv_vfmv_f_s_f32m1_f32(vmax1);

      if (vmax > max_dist) {
        const vbool16_t m = __riscv_vmfeq_vf_f32m2_b16(vdist2, vmax, vl);
        const long lane = __riscv_vfirst_m_b16(m, vl);
        if (lane >= 0) {
          max_dist = vmax;
          i_min = i;
          i_max = j + static_cast<std::size_t>(lane);
        }
      }

      j += vl;
    }
  }

  if (i_min == token || i_max == token)
    return {std::numeric_limits<float>::quiet_NaN(), token, token};

  return {max_dist, i_min, i_max};
}
#endif

int main(int argc, char** argv)
{
  std::size_t n = 2500;
  int iters = 20;
  int warmup = 3;
  if (argc >= 2) n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  if (argc >= 3) iters = std::max(1, std::atoi(argv[2]));
  if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));

  PointCloud cloud;
  fillRandomCloud(cloud, n, 7);

#if !defined(__RVV10__)
  std::cout << "[SKIP] __RVV10__ not enabled.\n";
  return 0;
#else
  // Correctness check (single run)
  const auto r_a = getMaxSegmentRVV_vlse(cloud);
  const auto r_b = getMaxSegmentRVV_vlsseg3(cloud);
  if (!(r_a.max_dist2 == r_b.max_dist2 && r_a.i_min == r_b.i_min && r_a.i_max == r_b.i_max)) {
    std::cerr << "[FAIL] Mismatch!\n"
              << "  vlse:    dist2=" << r_a.max_dist2 << " i_min=" << r_a.i_min << " i_max=" << r_a.i_max << "\n"
              << "  vlsseg3: dist2=" << r_b.max_dist2 << " i_min=" << r_b.i_min << " i_max=" << r_b.i_max << "\n";
    return 1;
  }

  auto run = [&](auto&& fn) -> double {
    for (int i = 0; i < warmup; ++i) {
      const auto r = fn(cloud);
      consume(r.max_dist2);
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      const auto r = fn(cloud);
      consume(r.max_dist2);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  g_sink_f32 = 0.0f;
  const double ms_vlse = run(getMaxSegmentRVV_vlse);
  const double ms_vlsseg3 = run(getMaxSegmentRVV_vlsseg3);
  const double speedup = (ms_vlsseg3 > 0.0) ? (ms_vlse / ms_vlsseg3) : 0.0;

  std::cout << "getMaxSegmentRVV Load Strategy Compare:\n";
  std::cout << "  n=" << n << " iters=" << iters << " warmup=" << warmup << "\n";
  std::cout << "  Mode A (3x vlse32):      " << std::fixed << std::setprecision(6) << ms_vlse << " ms/iter\n";
  std::cout << "  Mode B (vlsseg3e32):     " << std::fixed << std::setprecision(6) << ms_vlsseg3 << " ms/iter\n";
  std::cout << "  Speedup (A/B):           " << std::fixed << std::setprecision(3) << speedup << "x\n";
  std::cout << "  check: dist2=" << r_a.max_dist2 << " i_min=" << r_a.i_min << " i_max=" << r_a.i_max << "\n";
  // std::cout << "  sink=" << g_sink_f32 << "\n";
  return 0;
#endif
}

