/**
 * RVV load micro-benchmarks (isolated).
 *
 * Purpose:
 * - Compare "3x scalar-field loads" vs "1x segmented tuple load" under different memory layouts.
 * - Keep loop structure simple so we primarily measure load traffic + address generation.
 *
 * Workloads / layout:
 * - Strided AoS: `PointT = pcl::PointXYZ` stored as an AoS array.
 *   Each lane loads one point, but fields are reached with a fixed byte stride (`sizeof(PointT)`).
 * - Contiguous (Packed xyz interleaved): `Packed3f {x,y,z}` stored as `xyzxyzxyz...` with 12-byte stride.
 *   This isolates the "packed interleaved" case where segment loads/stores are often strongest.
 * - Indexed AoS gather: random indices into the AoS array (indirect / non-contiguous access).
 *
 * Run/print order is aligned with the store benchmark:
 *   1) Strided AoS
 *   2) Contiguous (Packed xyz interleaved)
 *   3) Indexed AoS gather
 *
 * Metrics (letters match the final report; A/B/C/D... follow print order):
 * - [Strided AoS]
 *   - A: `strided_load3_fields_f32m2` → 3×`vlse32`
 *   - B: `strided_load3_seg_f32m2` → `vlsseg3e32`
 * - [Contiguous (Packed xyz interleaved)]
 *   - C: 3×`vlse32` via `strided_load_f32m2` (stride=12)
 *   - D: `contiguous_seg3_load_f32m2` → `vlseg3e32`
 * - [Indexed AoS gather]
 *   - E: `indexed_load3_fields_f32m2` → 3×`vluxei32`
 *   - F: `indexed_load3_seg_f32m2` → `vluxseg3ei32`
 *
 * Notes:
 * - A/B use the same index stream; C/D use the same AoS data; E/F use the same packed stream.
 * - We reduce the loaded vectors into a scalar sink to prevent dead-code elimination.
 */
#include <pcl/point_types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#if defined(__RVV10__)
#include <riscv_vector.h>
// #include "rvv_point_load.hpp" // original local wrapper (kept for reference)
// // Alias to avoid spelling the provider namespace at every call site.
// namespace rvv_load = ::rvv_test::rvv_load;

#include <pcl/common/rvv_point_load.h>

// Alias to avoid spelling the provider namespace at every call site.
namespace rvv_load = ::pcl::rvv_load;
#endif

using PointT = pcl::PointXYZ;

struct Packed3f {
  float x;
  float y;
  float z;
};

static volatile float g_sink_f32 = 0.0f;
static inline void consume(float v) { g_sink_f32 += v * 1.0000001f; }
static inline void logProgress(const char* msg)
{
  std::cout << msg << std::endl;
  std::cout.flush();
}

static void fillRandomPoints(std::vector<PointT>& pts, std::size_t n, std::uint32_t seed = 42)
{
  pts.resize(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-10.0f, 10.0f);
  for (std::size_t i = 0; i < n; ++i) {
    pts[i].x = u(rng);
    pts[i].y = u(rng);
    pts[i].z = u(rng);
  }
}

static void fillRandomIndices(std::vector<std::uint32_t>& idx, std::size_t n, std::size_t mod, std::uint32_t seed = 7)
{
  idx.resize(n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint32_t> u(0, static_cast<std::uint32_t>(mod - 1));
  for (std::size_t i = 0; i < n; ++i)
    idx[i] = u(rng);
}

#if defined(__RVV10__)
static inline float reduce3(const vfloat32m2_t a, const vfloat32m2_t b, const vfloat32m2_t c, std::size_t vl)
{
  // Sum a+b+c, then reduce-sum to scalar.
  const vfloat32m2_t s = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(a, b, vl), c, vl);
  const vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  // Some toolchains expose only the unsigned reduction intrinsic name.
  const vfloat32m1_t red = __riscv_vfredusum_vs_f32m2_f32m1(s, init, vl);
  return __riscv_vfmv_f_s_f32m1_f32(red);
}

static double bench_indexed_vluxei32(const std::vector<PointT>& pts,
                                     const std::vector<std::uint32_t>& idx,
                                     int iters,
                                     int warmup)
{
  const uint8_t* base = reinterpret_cast<const uint8_t*>(pts.data());
  const std::size_t n = idx.size();

  auto run_once = [&]() -> float {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n;) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
      const vuint32m2_t v_idx = __riscv_vle32_v_u32m2(idx.data() + i, vl);
      // 原本方式（保留用于对照）：
      // const vuint32m2_t v_off = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
      // const vfloat32m2_t vx =
      //     __riscv_vluxei32_v_f32m2(reinterpret_cast<const float*>(base + offsetof(PointT, x)), v_off, vl);
      // const vfloat32m2_t vy =
      //     __riscv_vluxei32_v_f32m2(reinterpret_cast<const float*>(base + offsetof(PointT, y)), v_off, vl);
      // const vfloat32m2_t vz =
      //     __riscv_vluxei32_v_f32m2(reinterpret_cast<const float*>(base + offsetof(PointT, z)), v_off, vl);

      // 原语路径：固定 3×`vluxei32`（`indexed_load3_fields_f32m2`）
      const vuint32m2_t v_off = rvv_load::byte_offsets_u32m2<PointT>(v_idx, vl);
      vfloat32m2_t vx, vy, vz;
      rvv_load::indexed_load3_fields_f32m2<PointT, offsetof(PointT, x), offsetof(PointT, y), offsetof(PointT, z)>(
          base, v_off, vl, vx, vy, vz);
      acc += reduce3(vx, vy, vz, vl);
      i += vl;
    }
    return acc;
  };

  for (int i = 0; i < warmup; ++i) consume(run_once());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    consume(run_once());
    if ((i + 1) % 50 == 0) {
      std::cout << "    progress: " << (i + 1) << "/" << iters << std::endl;
      std::cout.flush();
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double bench_indexed_vluxseg3ei32(const std::vector<PointT>& pts,
                                         const std::vector<std::uint32_t>& idx,
                                         int iters,
                                         int warmup)
{
  const uint8_t* base = reinterpret_cast<const uint8_t*>(pts.data());
  const std::size_t n = idx.size();

  auto run_once = [&]() -> float {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n;) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
      const vuint32m2_t v_idx = __riscv_vle32_v_u32m2(idx.data() + i, vl);
      const vuint32m2_t v_off = rvv_load::byte_offsets_u32m2<PointT>(v_idx, vl);
      // 原本方式（保留用于对照）：
      // const vfloat32m2x3_t v_xyz =
      //     __riscv_vluxseg3ei32_v_f32m2x3(reinterpret_cast<const float*>(base + offsetof(PointT, x)), v_off, vl);
      // const vfloat32m2_t vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
      // const vfloat32m2_t vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
      // const vfloat32m2_t vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

      // 原语路径：固定 `vluxseg3ei32`（调用方保证 x/y/z 在结构体内连续）
      vfloat32m2_t vx, vy, vz;
      rvv_load::indexed_load3_seg_f32m2(reinterpret_cast<const float*>(base + offsetof(PointT, x)), v_off, vl, vx, vy, vz);
      acc += reduce3(vx, vy, vz, vl);
      i += vl;
    }
    return acc;
  };

  for (int i = 0; i < warmup; ++i) consume(run_once());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    consume(run_once());
    if ((i + 1) % 50 == 0) {
      std::cout << "    progress: " << (i + 1) << "/" << iters << std::endl;
      std::cout.flush();
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double bench_strided_vlse32(const std::vector<PointT>& pts, int iters, int warmup)
{
  const std::size_t n = pts.size();

  auto run_once = [&]() -> float {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n;) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
      // 原本方式（保留用于对照）：
      // const float* xptr = &pts[i].x;
      // const vfloat32m2_t vx = __riscv_vlse32_v_f32m2(xptr, stride, vl);
      // ... y, z 同理

      // 原语路径：固定 3×`vlse32`（`strided_load3_fields_f32m2`）
      const std::uint8_t* base_u8 = reinterpret_cast<const std::uint8_t*>(&pts[i]);
      vfloat32m2_t vx, vy, vz;
      rvv_load::strided_load3_fields_f32m2<sizeof(PointT), offsetof(PointT, x), offsetof(PointT, y), offsetof(PointT, z)>(
          base_u8, vl, vx, vy, vz);
      acc += reduce3(vx, vy, vz, vl);
      i += vl;
    }
    return acc;
  };

  for (int i = 0; i < warmup; ++i) consume(run_once());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    consume(run_once());
    if ((i + 1) % 50 == 0) {
      std::cout << "    progress: " << (i + 1) << "/" << iters << std::endl;
      std::cout.flush();
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double bench_strided_vlsseg3e32(const std::vector<PointT>& pts, int iters, int warmup)
{
  const std::size_t n = pts.size();

  auto run_once = [&]() -> float {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n;) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
      // 原本方式（保留用于对照）：
      // const float* base = &pts[i].x;
      // const vfloat32m2x3_t v_xyz = __riscv_vlsseg3e32_v_f32m2x3(base, stride, vl);
      // const vfloat32m2_t vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
      // const vfloat32m2_t vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
      // const vfloat32m2_t vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

      // 原语路径：固定 `vlsseg3e32`（`strided_load3_seg_f32m2`；首字段指针指向连续 xyz 的首元素）
      vfloat32m2_t vx, vy, vz;
      rvv_load::strided_load3_seg_f32m2<sizeof(PointT)>(&pts[i].x, vl, vx, vy, vz);
      acc += reduce3(vx, vy, vz, vl);
      i += vl;
    }
    return acc;
  };

  for (int i = 0; i < warmup; ++i) consume(run_once());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    consume(run_once());
    if ((i + 1) % 50 == 0) {
      std::cout << "    progress: " << (i + 1) << "/" << iters << std::endl;
      std::cout.flush();
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double bench_packed_vlse32(const std::vector<Packed3f>& pts, int iters, int warmup)
{
  const std::size_t n = pts.size();
  constexpr ptrdiff_t stride = static_cast<ptrdiff_t>(sizeof(Packed3f)); // 12 bytes

  auto run_once = [&]() -> float {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n;) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
      const float* xptr = &pts[i].x;
      const float* yptr = &pts[i].y;
      const float* zptr = &pts[i].z;
      // 原本方式（保留用于对照）：
      // const vfloat32m2_t vx = __riscv_vlse32_v_f32m2(xptr, stride, vl);
      // const vfloat32m2_t vy = __riscv_vlse32_v_f32m2(yptr, stride, vl);
      // const vfloat32m2_t vz = __riscv_vlse32_v_f32m2(zptr, stride, vl);

      // 抽象后的方式（等价的 3x vlse32；此处 stride=12）：
      const vfloat32m2_t vx = rvv_load::strided_load_f32m2<sizeof(Packed3f)>(xptr, vl);
      const vfloat32m2_t vy = rvv_load::strided_load_f32m2<sizeof(Packed3f)>(yptr, vl);
      const vfloat32m2_t vz = rvv_load::strided_load_f32m2<sizeof(Packed3f)>(zptr, vl);
      acc += reduce3(vx, vy, vz, vl);
      i += vl;
    }
    return acc;
  };

  for (int i = 0; i < warmup; ++i) consume(run_once());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    consume(run_once());
    if ((i + 1) % 50 == 0) {
      std::cout << "    progress: " << (i + 1) << "/" << iters << std::endl;
      std::cout.flush();
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double bench_packed_vlseg3e32(const std::vector<Packed3f>& pts, int iters, int warmup)
{
  const std::size_t n = pts.size();

  auto run_once = [&]() -> float {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n;) {
      const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
      // 原本方式（保留用于对照）：
      // const float* base = &pts[i].x;
      // const vfloat32m2x3_t v_xyz = __riscv_vlseg3e32_v_f32m2x3(base, vl);
      // const vfloat32m2_t vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
      // const vfloat32m2_t vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
      // const vfloat32m2_t vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

      // 抽象后的方式（等价的 vlseg3e32）：
      const float* base = &pts[i].x;
      vfloat32m2_t vx, vy, vz;
      rvv_load::contiguous_seg3_load_f32m2(base, vl, vx, vy, vz);
      acc += reduce3(vx, vy, vz, vl);
      i += vl;
    }
    return acc;
  };

  for (int i = 0; i < warmup; ++i) consume(run_once());
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    consume(run_once());
    if ((i + 1) % 50 == 0) {
      std::cout << "    progress: " << (i + 1) << "/" << iters << std::endl;
      std::cout.flush();
    }
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}
#endif

int main(int argc, char** argv)
{
  std::size_t n_points = 1u << 20; // 1,048,576
  int iters = 200;
  int warmup = 20;
  if (argc >= 2) n_points = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  if (argc >= 3) iters = std::max(1, std::atoi(argv[2]));
  if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));

  std::vector<PointT> pts;
  fillRandomPoints(pts, n_points, 42);

  // For indexed tests, use a smaller index stream to emulate typical indirect access.
  std::vector<std::uint32_t> idx;
  fillRandomIndices(idx, n_points, n_points, 7);

  // Packed AoS stream for contiguous segment-load (vlseg3e32).
  std::vector<Packed3f> packed(n_points);
  for (std::size_t i = 0; i < n_points; ++i) {
    packed[i].x = pts[i].x;
    packed[i].y = pts[i].y;
    packed[i].z = pts[i].z;
  }

#if !defined(__RVV10__)
  std::cout << "[SKIP] __RVV10__ not enabled.\n";
  return 0;
#else
  g_sink_f32 = 0.0f;

  logProgress("RVV Load MicroBench: starting...");
  std::cout << "  n_points=" << n_points << " iters=" << iters << " warmup=" << warmup << std::endl;
  std::cout.flush();

  // Keep the run / report order aligned with store benchmark: Stride -> Contiguous(Packed) -> Indexed.
  logProgress("  [1/6] Strided AoS: 3x vlse32");
  const double t_vlse = bench_strided_vlse32(pts, iters, warmup);
  logProgress("  [2/6] Strided AoS: vlsseg3e32");
  const double t_vlsseg3 = bench_strided_vlsseg3e32(pts, iters, warmup);

  // Packed AoS (xyz interleaved): compare 3x vlse32(stride=12) vs 1x vlseg3e32
  logProgress("  [3/6] Contiguous (Packed xyz interleaved): 3x vlse32 (stride=12)");
  const double t_packed_vlse = bench_packed_vlse32(packed, iters, warmup);
  logProgress("  [4/6] Contiguous (Packed xyz interleaved): vlseg3e32");
  const double t_packed_vlseg3 = bench_packed_vlseg3e32(packed, iters, warmup);

  logProgress("  [5/6] Indexed AoS gather: 3x vluxei32");
  const double t_vlux = bench_indexed_vluxei32(pts, idx, iters, warmup);
  logProgress("  [6/6] Indexed AoS gather: vluxseg3ei32");
  const double t_vluxseg3 = bench_indexed_vluxseg3ei32(pts, idx, iters, warmup);

  std::cout << "RVV Load MicroBench:\n";
  std::cout << "  n_points=" << n_points << " iters=" << iters << " warmup=" << warmup << "\n";
  std::cout << "  [Strided AoS]\n";
  std::cout << "    A: 3x vlse32        " << std::fixed << std::setprecision(6) << t_vlse << " ms/iter\n";
  std::cout << "    B: vlsseg3e32       " << std::fixed << std::setprecision(6) << t_vlsseg3 << " ms/iter\n";
  std::cout << "    speedup (A/B):      " << std::fixed << std::setprecision(3) << (t_vlse / t_vlsseg3) << "x\n";
  std::cout << "  [Contiguous (Packed xyz interleaved)]\n";
  std::cout << "    C: 3x vlse32        " << std::fixed << std::setprecision(6) << t_packed_vlse << " ms/iter\n";
  std::cout << "    D: vlseg3e32        " << std::fixed << std::setprecision(6) << t_packed_vlseg3 << " ms/iter\n";
  std::cout << "    speedup (C/D):      " << std::fixed << std::setprecision(3) << (t_packed_vlse / t_packed_vlseg3) << "x\n";
  std::cout << "  [Indexed AoS gather]\n";
  std::cout << "    E: 3x vluxei32      " << std::fixed << std::setprecision(6) << t_vlux << " ms/iter\n";
  std::cout << "    F: vluxseg3ei32     " << std::fixed << std::setprecision(6) << t_vluxseg3 << " ms/iter\n";
  std::cout << "    speedup (E/F):      " << std::fixed << std::setprecision(3) << (t_vlux / t_vluxseg3) << "x\n";
  // std::cout << "  sink=" << g_sink_f32 << "\n";
  return 0;
#endif
}

