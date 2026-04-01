/**
 * Micro-benchmark for comparing 4x strided scalar stores vs a 4-segment store.
 *
 * Stores target the layout of `pcl::PointXYZIEdge`:
 *   magnitude      (segment 0)
 *   direction      (segment 1)
 *   magnitude_x    (segment 2)
 *   magnitude_y    (segment 3)
 *
 * The store methods compared:
 *   - A: 4x __riscv_vsse32_v_f32m2 (one per field)
 *   - B:   __riscv_vssseg4e32_v_f32m2x4 (one segment-store for the 4 fields)
 *
 * Note: We keep the load+extract path identical across both modes by using:
 *   vlsseg4e32 (tuple load) in both cases, then:
 *     - Mode A extracts vt segments and uses vsse32 for each field
 *     - Mode B directly uses vssseg4 to write all 4 segments at once
 */
// Keep this micro-benchmark self-contained:
// We define a local struct that matches the memory layout of PCL's
// `pcl::PointXYZIEdge`:
//   - 16-byte aligned XYZ padding via `PCL_ADD_POINT4D` (float data[4])
//   - then 4 consecutive float fields:
//       magnitude, direction, magnitude_x, magnitude_y
//
// This avoids relying on PCL headers for linter/include-path resolution.
struct alignas(16) PointXYZIEdgeTest {
  union {
    float data[4];
    struct {
      float x;
      float y;
      float z;
    };
  };

  float magnitude;
  float direction;
  float magnitude_x;
  float magnitude_y;
};

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#if defined(__RVV10__)
#include <riscv_vector.h>
#endif

static volatile float g_sink_f32 = 0.0f;
static inline void consumeFloat(float v)
{
  g_sink_f32 += v * 1.0000001f;
}

template <typename PointT>
struct EdgeStoreLayout {
  static constexpr std::size_t stride_bytes = sizeof(PointT);
  static constexpr std::size_t stride_f32 = stride_bytes / sizeof(float);

  static constexpr std::size_t off_mag = offsetof(PointT, magnitude);
  static constexpr std::size_t off_dir = offsetof(PointT, direction);
  static constexpr std::size_t off_mx = offsetof(PointT, magnitude_x);
  static constexpr std::size_t off_my = offsetof(PointT, magnitude_y);

  static constexpr std::size_t off_seg0_f32 = off_mag / sizeof(float);
  static constexpr std::size_t off_seg1_f32 = off_dir / sizeof(float);
  static constexpr std::size_t off_seg2_f32 = off_mx / sizeof(float);
  static constexpr std::size_t off_seg3_f32 = off_my / sizeof(float);

  static_assert(stride_bytes % sizeof(float) == 0, "stride must be multiple of 4 bytes");
  static_assert(off_dir - off_mag == sizeof(float),
                "Expected direction to be 4 bytes after magnitude");
  static_assert(off_mx - off_dir == sizeof(float),
                "Expected magnitude_x to be 4 bytes after direction");
  static_assert(off_my - off_mx == sizeof(float),
                "Expected magnitude_y to be 4 bytes after magnitude_x");
};

int main(int argc, char** argv)
{
#if !defined(__RVV10__)
  std::cerr << "This test requires __RVV10__ (USE_PCL_RVV10=1)." << std::endl;
  return 2;
#else
  using PointT = PointXYZIEdgeTest;
  using L = EdgeStoreLayout<PointT>;
  constexpr std::size_t n_default = 640u * 480u;
  constexpr int iters_default = 20;
  constexpr int warmup_default = 5;

  std::size_t n = n_default;
  int iterations = iters_default;
  int warmup = warmup_default;

  if (argc >= 2) n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  if (argc >= 3) iterations = std::atoi(argv[2]);
  if (argc >= 4) warmup = std::atoi(argv[3]);

  // Source layout mimics `PointXYZIEdge` fields inside the point-stride memory.
  // We only store the 4 float fields; the remaining bytes per-point are left as padding.
  std::vector<float> src(n * L::stride_f32, 0.0f);
  std::vector<float> dst(n * L::stride_f32, 0.0f);

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t base = i * L::stride_f32;
    src[base + L::off_seg0_f32] = static_cast<float>(i) + 1.0f; // magnitude
    src[base + L::off_seg1_f32] = static_cast<float>(i) + 2.0f; // direction
    src[base + L::off_seg2_f32] = static_cast<float>(i) + 3.0f; // magnitude_x
    src[base + L::off_seg3_f32] = static_cast<float>(i) + 4.0f; // magnitude_y
  }

  auto run_mode_vsse = [&]() -> double {
    std::fill(dst.begin(), dst.end(), 0.0f);
    std::size_t j0 = 0;
    double total_ns = 0.0;

    for (int it = -warmup; it < iterations; ++it) {
      auto t_start = std::chrono::high_resolution_clock::now();

      j0 = 0;
      while (j0 < n) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(n - j0));

        const float* seg_base_in =
            src.data() + j0 * L::stride_f32 + L::off_seg0_f32;
        vfloat32m2x4_t vt = __riscv_vlsseg4e32_v_f32m2x4(seg_base_in, L::stride_bytes, vl);

        vfloat32m2_t v_mag = __riscv_vget_v_f32m2x4_f32m2(vt, 0);
        vfloat32m2_t v_dir = __riscv_vget_v_f32m2x4_f32m2(vt, 1);
        vfloat32m2_t v_mx = __riscv_vget_v_f32m2x4_f32m2(vt, 2);
        vfloat32m2_t v_my = __riscv_vget_v_f32m2x4_f32m2(vt, 3);

        float* out_mag =
            dst.data() + j0 * L::stride_f32 + L::off_seg0_f32;
        float* out_dir =
            dst.data() + j0 * L::stride_f32 + L::off_seg1_f32;
        float* out_mx =
            dst.data() + j0 * L::stride_f32 + L::off_seg2_f32;
        float* out_my =
            dst.data() + j0 * L::stride_f32 + L::off_seg3_f32;

        __riscv_vsse32_v_f32m2(out_mag, L::stride_bytes, v_mag, vl);
        __riscv_vsse32_v_f32m2(out_dir, L::stride_bytes, v_dir, vl);
        __riscv_vsse32_v_f32m2(out_mx, L::stride_bytes, v_mx, vl);
        __riscv_vsse32_v_f32m2(out_my, L::stride_bytes, v_my, vl);

        j0 += vl;
      }

      auto t_end = std::chrono::high_resolution_clock::now();
      if (it >= 0)
        total_ns += static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
                .count());

      // Prevent the whole loop from being optimized away.
      consumeFloat(dst[0]);
      consumeFloat(dst[n / 2 * L::stride_f32 + L::off_seg0_f32]);
      consumeFloat(dst[(n - 1) * L::stride_f32 + L::off_seg3_f32]);
    }

    // Sanity check (exact bitwise equality, sampled).
    bool ok = true;
    const std::size_t step = std::max<std::size_t>(1, n / 1024);
    for (std::size_t i = 0; i < n; i += step) {
      const std::size_t base = i * L::stride_f32;
      if (dst[base + L::off_seg0_f32] != src[base + L::off_seg0_f32]) ok = false;
      if (dst[base + L::off_seg1_f32] != src[base + L::off_seg1_f32]) ok = false;
      if (dst[base + L::off_seg2_f32] != src[base + L::off_seg2_f32]) ok = false;
      if (dst[base + L::off_seg3_f32] != src[base + L::off_seg3_f32]) ok = false;
    }
    if (!ok) {
      std::cerr << "[EdgeStoreBench] Mode A correctness check FAILED." << std::endl;
      return -1.0;
    }

    return total_ns / iterations;
  };

  auto run_mode_vssseg4 = [&]() -> double {
    std::fill(dst.begin(), dst.end(), 0.0f);
    std::size_t j0 = 0;
    double total_ns = 0.0;

    for (int it = -warmup; it < iterations; ++it) {
      auto t_start = std::chrono::high_resolution_clock::now();

      j0 = 0;
      while (j0 < n) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(n - j0));

        const float* seg_base_in =
            src.data() + j0 * L::stride_f32 + L::off_seg0_f32;
        vfloat32m2x4_t vt = __riscv_vlsseg4e32_v_f32m2x4(seg_base_in, L::stride_bytes, vl);

        float* seg_base_out =
            dst.data() + j0 * L::stride_f32 + L::off_seg0_f32;
        __riscv_vssseg4e32_v_f32m2x4(seg_base_out, L::stride_bytes, vt, vl);

        j0 += vl;
      }

      auto t_end = std::chrono::high_resolution_clock::now();
      if (it >= 0)
        total_ns += static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
                .count());

      // Prevent DCE.
      consumeFloat(dst[0]);
      consumeFloat(dst[n / 2 * L::stride_f32 + L::off_seg0_f32]);
      consumeFloat(dst[(n - 1) * L::stride_f32 + L::off_seg3_f32]);
    }

    // Sanity check (exact bitwise equality, sampled).
    bool ok = true;
    const std::size_t step = std::max<std::size_t>(1, n / 1024);
    for (std::size_t i = 0; i < n; i += step) {
      const std::size_t base = i * L::stride_f32;
      if (dst[base + L::off_seg0_f32] != src[base + L::off_seg0_f32]) ok = false;
      if (dst[base + L::off_seg1_f32] != src[base + L::off_seg1_f32]) ok = false;
      if (dst[base + L::off_seg2_f32] != src[base + L::off_seg2_f32]) ok = false;
      if (dst[base + L::off_seg3_f32] != src[base + L::off_seg3_f32]) ok = false;
    }
    if (!ok) {
      std::cerr << "[EdgeStoreBench] Mode B correctness check FAILED." << std::endl;
      return -1.0;
    }

    return total_ns / iterations;
  };

  // For repeatability on board: reset sink once.
  g_sink_f32 = 0.0f;

  const double avg_ns_vsse = run_mode_vsse();
  const double avg_ns_vssseg4 = run_mode_vssseg4();
  if (avg_ns_vsse < 0.0 || avg_ns_vssseg4 < 0.0) return 1;

  std::cout << "EdgeStoreBench (PointXYZIEdge layout):" << std::endl;
  std::cout << "  n=" << n << ", iterations=" << iterations << ", warmup=" << warmup << std::endl;
  std::cout << "  stride_bytes=" << L::stride_bytes << " (float_stride=" << L::stride_f32 << ")" << std::endl;
  std::cout << "  offsets_f32: mag=" << L::off_seg0_f32 << ", dir=" << L::off_seg1_f32
            << ", mx=" << L::off_seg2_f32 << ", my=" << L::off_seg3_f32 << std::endl;

  std::cout << "  Mode A (4x vsse32):   " << (avg_ns_vsse / 1e6) << " ms/iter" << std::endl;
  std::cout << "  Mode B (vssseg4e32): " << (avg_ns_vssseg4 / 1e6) << " ms/iter" << std::endl;
  if (avg_ns_vsse > 0.0) {
    const double speedup = avg_ns_vsse / avg_ns_vssseg4;
    std::cout << "  Speedup (A/B): " << speedup << "x" << std::endl;
  }

  // Ensure sink is observable.
  std::cout << "  sink=" << g_sink_f32 << std::endl;
  return 0;
#endif
}

