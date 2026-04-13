/**
 * RVV store micro-benchmark (isolated).
 *
 * Purpose:
 * - Compare "4x scalar-field stores" vs "1x segmented tuple store" under different store patterns.
 * - Keep inputs constant (no loads) so we primarily measure store traffic + address generation.
 *
 * Workloads / layout:
 * - Strided AoS: store into an AoS array `dst[]` with a fixed byte stride (`sizeof(Edge4f)`).
 *   Fields `f0..f3` are consecutive floats in this struct, mimicking the common "payload + padding" layout.
 * - Contiguous: store into 4 independent float arrays (`dst_c0..dst_c3`) to model SoA writes.
 * - Indexed scatter: store into AoS using a random index stream (indirect / non-contiguous stores).
 *
 * Run/print order matches `bench_rvv_load_compare.cpp`: strided → contiguous → indexed,
 * for both **4-field** (e.g. edge payload) and **3-field xyz** (e.g. `common.hpp` distances).
 *
 * Metrics (A–F: four fields; G–L: three fields):
 * - [Strided AoS — 4 floats]
 *   - A: `strided_store4_fields_f32m2` → 4×`vsse32`
 *   - B: `strided_store4_seg_f32m2` → `vssseg4e32`
 * - [Contiguous — 4]
 *   - C: 4× `vse32` / `contiguous_store4_f32m2`
 *   - D: `vsseg4e32` / `contiguous_seg4_store_f32m2`
 * - [Indexed scatter — 4]
 *   - E: `scatter_store4_fields_f32m2` → 4×`vsuxei32`
 *   - F: `scatter_store4_seg_f32m2` → `vsuxseg4ei32`
 * - [Strided AoS — 3 floats xyz]
 *   - G: `strided_store3_fields_f32m2` → 3×`vsse32`
 *   - H: `strided_store3_seg_f32m2` → `vssseg3e32`
 * - [Contiguous — 3]
 *   - I: 3× `vse32` / `contiguous_store3_f32m2`
 *   - J: `vsseg3e32` / `contiguous_seg3_store_f32m2`
 * - [Indexed scatter — 3]
 *   - K: `scatter_store3_fields_f32m2` → 3×`vsuxei32`
 *   - L: `scatter_store3_seg_f32m2` → `vsuxseg3ei32`
 *
 * Notes:
 * - In the refactor, store helpers may automatically fall back to 4x scalar stores
 *   if the fields are not tightly packed; here `Edge4f::f0..f3` are consecutive so
 *   the segmented form is expected to be usable.
 */
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#if defined(__RVV10__)
#include <riscv_vector.h>
// #include "rvv_point_store.hpp" // original local wrapper (kept for reference)
// Alias to avoid spelling the provider namespace at every call site.
// namespace rvv_store = ::rvv_test::rvv_store;

#include <pcl/common/rvv_point_store.h>

// Alias to avoid spelling the provider namespace at every call site.
namespace rvv_store = ::pcl::rvv_store;
#endif

static volatile float g_sink_f32 = 0.0f;
static inline void consume(float v) { g_sink_f32 += v * 1.0000001f; }
static inline void logProgress(const char* msg)
{
  std::cout << msg << std::endl;
  std::cout.flush();
}

// A minimal AoS layout with 4 consecutive float fields to store.
struct alignas(16) Edge4f {
  float pad[4]; // mimic PCL_ADD_POINT4D space (unused)
  float f0;
  float f1;
  float f2;
  float f3;
};

// Three consecutive floats after padding (mimic xyz writeback after PCL_ADD_POINT4D).
struct alignas(16) Edge3f {
  float pad[4];
  float x;
  float y;
  float z;
};

struct Packed3f {
  float x;
  float y;
  float z;
};

#if defined(__RVV10__)
static void fillIndices(std::vector<std::uint32_t>& idx, std::size_t n, std::uint32_t seed = 7)
{
  idx.resize(n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint32_t> u(0, static_cast<std::uint32_t>(n - 1));
  for (std::size_t i = 0; i < n; ++i) idx[i] = u(rng);
}
#endif

int main(int argc, char** argv)
{
  std::size_t n = 1u << 20; // 1,048,576
  int iters = 200;
  int warmup = 20;
  if (argc >= 2) n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  if (argc >= 3) iters = std::max(1, std::atoi(argv[2]));
  if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));

  std::vector<Edge4f> dst(n);
  std::vector<Edge3f> dst3(n);
  std::vector<float> dst_c0(n), dst_c1(n), dst_c2(n), dst_c3(n);
  std::vector<float> d3_c0(n), d3_c1(n), d3_c2(n);
  std::vector<Packed3f> packed3(n);
  std::vector<std::uint32_t> idx;

#if !defined(__RVV10__)
  std::cout << "[SKIP] __RVV10__ not enabled.\n";
  return 0;
#else
  // Indexed scatter indices (random).
  fillIndices(idx, n, 7);

  constexpr ptrdiff_t stride_bytes = static_cast<ptrdiff_t>(sizeof(Edge4f));
  constexpr std::size_t off0 = offsetof(Edge4f, f0);
  constexpr std::size_t off1 = offsetof(Edge4f, f1);
  constexpr std::size_t off2 = offsetof(Edge4f, f2);
  constexpr std::size_t off3 = offsetof(Edge4f, f3);

  constexpr ptrdiff_t stride3_bytes = static_cast<ptrdiff_t>(sizeof(Edge3f));
  constexpr std::size_t ox = offsetof(Edge3f, x);
  constexpr std::size_t oy = offsetof(Edge3f, y);
  constexpr std::size_t oz = offsetof(Edge3f, z);

  auto run_vsse = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);

        uint8_t* base = reinterpret_cast<uint8_t*>(dst.data() + i);
        rvv_store::strided_store4_fields_f32m2<sizeof(Edge4f), off0, off1, off2, off3>(
            base, vl, v0, v1, v2, v3);
        i += vl;
      }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);

        uint8_t* base = reinterpret_cast<uint8_t*>(dst.data() + i);
        rvv_store::strided_store4_fields_f32m2<sizeof(Edge4f), off0, off1, off2, off3>(
            base, vl, v0, v1, v2, v3);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    // Touch a few elements to keep stores observable.
    consume(dst[0].f0);
    consume(dst[n / 2].f1);
    consume(dst[n - 1].f3);

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vssseg4 = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
        rvv_store::strided_store4_seg_f32m2<sizeof(Edge4f)>(&(dst.data() + i)->f0, vl, v0, v1, v2, v3);
        i += vl;
      }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
        rvv_store::strided_store4_seg_f32m2<sizeof(Edge4f)>(&(dst.data() + i)->f0, vl, v0, v1, v2, v3);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    consume(dst[0].f0);
    consume(dst[n / 2].f2);
    consume(dst[n - 1].f3);

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vse4 = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
        // 原本方式（保留用于对照）：
        // __riscv_vse32_v_f32m2(dst_c0.data() + i, v0, vl);
        // __riscv_vse32_v_f32m2(dst_c1.data() + i, v1, vl);
        // __riscv_vse32_v_f32m2(dst_c2.data() + i, v2, vl);
        // __riscv_vse32_v_f32m2(dst_c3.data() + i, v3, vl);

        rvv_store::contiguous_store4_f32m2(dst_c0.data() + i,
                                           dst_c1.data() + i,
                                           dst_c2.data() + i,
                                           dst_c3.data() + i,
                                           vl, v0, v1, v2, v3);
        i += vl;
      }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
        rvv_store::contiguous_store4_f32m2(dst_c0.data() + i,
                                           dst_c1.data() + i,
                                           dst_c2.data() + i,
                                           dst_c3.data() + i,
                                           vl, v0, v1, v2, v3);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    consume(dst_c0[0]);
    consume(dst_c1[n / 2]);
    consume(dst_c3[n - 1]);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vsseg4 = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
        // 原本方式（保留用于对照）：
        // vfloat32m2x4_t vt = __riscv_vset_v_f32m2_f32m2x4(__riscv_vundefined_f32m2x4(), 0, v0);
        // vt = __riscv_vset_v_f32m2_f32m2x4(vt, 1, v1);
        // vt = __riscv_vset_v_f32m2_f32m2x4(vt, 2, v2);
        // vt = __riscv_vset_v_f32m2_f32m2x4(vt, 3, v3);
        // float* base = dst_c0.data() + i;
        // __riscv_vsseg4e32_v_f32m2x4(base, vt, vl);

        rvv_store::contiguous_seg4_store_f32m2(dst_c0.data() + i, vl, v0, v1, v2, v3);
        i += vl;
      }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
        rvv_store::contiguous_seg4_store_f32m2(dst_c0.data() + i, vl, v0, v1, v2, v3);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    consume(dst_c0[0]);
    consume(dst_c0[n / 2]);
    consume(dst_c0[n - 1]);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vsux_scatter = [&]() -> std::pair<double, double> {
    // Returns {E_time, F_time}: scatter_store4_fields vs scatter_store4_seg (primitives).
    g_sink_f32 = 0.0f;
    const uint8_t* base_u8 = reinterpret_cast<const uint8_t*>(dst.data());
    const std::size_t n_idx = idx.size();

    auto run_mode_a = [&](bool measure) -> double {
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < (measure ? iters : warmup); ++it) {
        for (std::size_t i = 0; i < n_idx;) {
          const std::size_t vl = __riscv_vsetvl_e32m2(n_idx - i);
          const vuint32m2_t v_idx = __riscv_vle32_v_u32m2(idx.data() + i, vl);
          const vuint32m2_t v_off = rvv_store::byte_offsets_u32m2<Edge4f>(v_idx, vl);
          const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
          const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
          const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
          const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
          rvv_store::scatter_store4_fields_f32m2<off0, off1, off2, off3>(
              const_cast<std::uint8_t*>(base_u8), v_off, vl, v0, v1, v2, v3);
          i += vl;
        }
        if (measure && (it + 1) % 50 == 0) {
          std::cout << "    progress: " << (it + 1) << "/" << iters << std::endl;
          std::cout.flush();
        }
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      if (!measure) return 0.0;
      consume(dst[0].f0);
      consume(dst[n / 2].f2);
      consume(dst[n - 1].f3);
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    auto run_mode_b = [&](bool measure) -> double {
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < (measure ? iters : warmup); ++it) {
        for (std::size_t i = 0; i < n_idx;) {
          const std::size_t vl = __riscv_vsetvl_e32m2(n_idx - i);
          const vuint32m2_t v_idx = __riscv_vle32_v_u32m2(idx.data() + i, vl);
          const vuint32m2_t v_off = rvv_store::byte_offsets_u32m2<Edge4f>(v_idx, vl);
          const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
          const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
          const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
          const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
          float* seg_base = reinterpret_cast<float*>(const_cast<std::uint8_t*>(base_u8) + off0);
          rvv_store::scatter_store4_seg_f32m2(seg_base, v_off, vl, v0, v1, v2, v3);
          i += vl;
        }
        if (measure && (it + 1) % 50 == 0) {
          std::cout << "    progress: " << (it + 1) << "/" << iters << std::endl;
          std::cout.flush();
        }
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      if (!measure) return 0.0;
      consume(dst[0].f0);
      consume(dst[n / 2].f2);
      consume(dst[n - 1].f3);
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    // warmup both
    run_mode_a(false);
    run_mode_b(false);
    const double ta = run_mode_a(true);
    const double tb = run_mode_b(true);
    return {ta, tb};
  };

  auto run_strided_3x_vsse = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        uint8_t* base = reinterpret_cast<uint8_t*>(dst3.data() + i);
        rvv_store::strided_store3_fields_f32m2<sizeof(Edge3f), ox, oy, oz>(base, vl, vx, vy, vz);
        i += vl;
      }
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        uint8_t* base = reinterpret_cast<uint8_t*>(dst3.data() + i);
        rvv_store::strided_store3_fields_f32m2<sizeof(Edge3f), ox, oy, oz>(base, vl, vx, vy, vz);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress [3f stride]: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    consume(dst3[0].x);
    consume(dst3[n / 2].y);
    consume(dst3[n - 1].z);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_strided_vssseg3 = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        rvv_store::strided_store3_seg_f32m2<sizeof(Edge3f)>(&(dst3.data() + i)->x, vl, vx, vy, vz);
        i += vl;
      }
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        rvv_store::strided_store3_seg_f32m2<sizeof(Edge3f)>(&(dst3.data() + i)->x, vl, vx, vy, vz);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress [3f seg]: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    consume(dst3[0].x);
    consume(dst3[n / 2].y);
    consume(dst3[n - 1].z);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vse3 = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        rvv_store::contiguous_store3_f32m2(d3_c0.data() + i, d3_c1.data() + i, d3_c2.data() + i, vl, v0, v1, v2);
        i += vl;
      }
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        rvv_store::contiguous_store3_f32m2(d3_c0.data() + i, d3_c1.data() + i, d3_c2.data() + i, vl, v0, v1, v2);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress [3 SoA]: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    consume(d3_c0[0]);
    consume(d3_c1[n / 2]);
    consume(d3_c2[n - 1]);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vsseg3 = [&]() -> double {
    g_sink_f32 = 0.0f;
    for (int w = 0; w < warmup; ++w) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        rvv_store::contiguous_seg3_store_f32m2(&packed3[i].x, vl, v0, v1, v2);
        i += vl;
      }
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i < n;) {
        const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
        const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
        const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
        const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
        rvv_store::contiguous_seg3_store_f32m2(&packed3[i].x, vl, v0, v1, v2);
        i += vl;
      }
      if ((it + 1) % 50 == 0) {
        std::cout << "    progress [3 packed]: " << (it + 1) << "/" << iters << std::endl;
        std::cout.flush();
      }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    consume(packed3[0].x);
    consume(packed3[n / 2].y);
    consume(packed3[n - 1].z);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  auto run_vsux_scatter3 = [&]() -> std::pair<double, double> {
    g_sink_f32 = 0.0f;
    const uint8_t* base_u8 = reinterpret_cast<const uint8_t*>(dst3.data());
    const std::size_t n_idx = idx.size();

    auto run_mode_a = [&](bool measure) -> double {
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < (measure ? iters : warmup); ++it) {
        for (std::size_t i = 0; i < n_idx;) {
          const std::size_t vl = __riscv_vsetvl_e32m2(n_idx - i);
          const vuint32m2_t v_idx = __riscv_vle32_v_u32m2(idx.data() + i, vl);
          const vuint32m2_t v_off = rvv_store::byte_offsets_u32m2<Edge3f>(v_idx, vl);
          const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1.0f, vl);
          const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2.0f, vl);
          const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3.0f, vl);
          rvv_store::scatter_store3_fields_f32m2<ox, oy, oz>(const_cast<std::uint8_t*>(base_u8), v_off, vl, vx, vy, vz);
          i += vl;
        }
        if (measure && (it + 1) % 50 == 0) {
          std::cout << "    progress [3 scatter A]: " << (it + 1) << "/" << iters << std::endl;
          std::cout.flush();
        }
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      if (!measure) return 0.0;
      consume(dst3[0].x);
      consume(dst3[n / 2].y);
      consume(dst3[n - 1].z);
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    auto run_mode_b = [&](bool measure) -> double {
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < (measure ? iters : warmup); ++it) {
        for (std::size_t i = 0; i < n_idx;) {
          const std::size_t vl = __riscv_vsetvl_e32m2(n_idx - i);
          const vuint32m2_t v_idx = __riscv_vle32_v_u32m2(idx.data() + i, vl);
          const vuint32m2_t v_off = rvv_store::byte_offsets_u32m2<Edge3f>(v_idx, vl);
          const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1.0f, vl);
          const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2.0f, vl);
          const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3.0f, vl);
          float* seg_base = reinterpret_cast<float*>(const_cast<std::uint8_t*>(base_u8) + ox);
          rvv_store::scatter_store3_seg_f32m2(seg_base, v_off, vl, vx, vy, vz);
          i += vl;
        }
        if (measure && (it + 1) % 50 == 0) {
          std::cout << "    progress [3 scatter B]: " << (it + 1) << "/" << iters << std::endl;
          std::cout.flush();
        }
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      if (!measure) return 0.0;
      consume(dst3[0].x);
      consume(dst3[n / 2].y);
      consume(dst3[n - 1].z);
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    run_mode_a(false);
    run_mode_b(false);
    return {run_mode_a(true), run_mode_b(true)};
  };

  logProgress("RVV Store MicroBench: starting...");
  std::cout << "  n=" << n << " iters=" << iters << " warmup=" << warmup << std::endl;
  std::cout.flush();

  // Keep the run / report order aligned with load benchmark: Stride -> Contiguous -> Indexed.
  logProgress("  [1/12] Strided AoS (4f): strided_store4");
  const double t_a = run_vsse();
  logProgress("  [2/12] Strided AoS (4f): strided_store4_seg");
  const double t_b = run_vssseg4();
  logProgress("  [3/12] Contiguous (4f): 4x vse32");
  const double t_c = run_vse4();
  logProgress("  [4/12] Contiguous (4f): vsseg4e32");
  const double t_d = run_vsseg4();
  logProgress("  [5/12] Indexed scatter (4f): modes");
  const auto [t_e, t_f] = run_vsux_scatter(); // prints progress for mode A/B internally
  logProgress("  [6/12] Strided AoS (3f xyz): 3x vsse32");
  const double t_g = run_strided_3x_vsse();
  logProgress("  [7/12] Strided AoS (3f): strided_store3");
  const double t_h = run_strided_vssseg3();
  logProgress("  [8/12] Contiguous (3f): 3x vse32");
  const double t_i = run_vse3();
  logProgress("  [9/12] Contiguous (3f packed): vsseg3e32");
  const double t_j = run_vsseg3();
  logProgress("  [10/12] Indexed scatter (3f): modes");
  const auto [t_k, t_l] = run_vsux_scatter3();

  std::cout << "RVV Store MicroBench:\n";
  std::cout << "  n=" << n << " iters=" << iters << " warmup=" << warmup << "\n";
  std::cout << "  [Strided AoS — 4 fields]\n";
  std::cout << "    stride_bytes=" << stride_bytes << "\n";
  std::cout << "    A: strided_store4_fields (4×vsse32) " << std::fixed << std::setprecision(6) << t_a << " ms/iter\n";
  std::cout << "    B: strided_store4_seg (vssseg4)    " << std::fixed << std::setprecision(6) << t_b << " ms/iter\n";
  std::cout << "    speedup (A/B):      " << std::fixed << std::setprecision(3) << (t_a / t_b) << "x\n";
  std::cout << "  [Contiguous — 4]\n";
  std::cout << "    C: 4x vse32         " << std::fixed << std::setprecision(6) << t_c << " ms/iter\n";
  std::cout << "    D: vsseg4e32        " << std::fixed << std::setprecision(6) << t_d << " ms/iter\n";
  std::cout << "    speedup (C/D):      " << std::fixed << std::setprecision(3) << (t_c / t_d) << "x\n";

  std::cout << "  [Indexed scatter — 4]\n";
  std::cout << "    E: scatter_store4_fields (4×vsuxei32) " << std::fixed << std::setprecision(6) << t_e << " ms/iter\n";
  std::cout << "    F: scatter_store4_seg (vsuxseg4ei32) " << std::fixed << std::setprecision(6) << t_f << " ms/iter\n";
  std::cout << "    speedup (E/F):       " << std::fixed << std::setprecision(3) << (t_e / t_f) << "x\n";

  std::cout << "  [Strided AoS — 3 fields (xyz)]\n";
  std::cout << "    stride_bytes=" << stride3_bytes << "\n";
  std::cout << "    G: strided_store3_fields  " << std::fixed << std::setprecision(6) << t_g << " ms/iter\n";
  std::cout << "    H: strided_store3_seg     " << std::fixed << std::setprecision(6) << t_h << " ms/iter\n";
  std::cout << "    speedup (G/H):            " << std::fixed << std::setprecision(3) << (t_g / t_h) << "x\n";
  std::cout << "  [Contiguous — 3]\n";
  std::cout << "    I: 3× vse32               " << std::fixed << std::setprecision(6) << t_i << " ms/iter\n";
  std::cout << "    J: vsseg3e32 (packed xyz) " << std::fixed << std::setprecision(6) << t_j << " ms/iter\n";
  std::cout << "    speedup (I/J):            " << std::fixed << std::setprecision(3) << (t_i / t_j) << "x\n";
  std::cout << "  [Indexed scatter — 3]\n";
  std::cout << "    K: scatter_store3_fields  " << std::fixed << std::setprecision(6) << t_k << " ms/iter\n";
  std::cout << "    L: scatter_store3_seg     " << std::fixed << std::setprecision(6) << t_l << " ms/iter\n";
  std::cout << "    speedup (K/L):            " << std::fixed << std::setprecision(3) << (t_k / t_l) << "x\n";
  // std::cout << "  sink=" << g_sink_f32 << "\n";
  return 0;
#endif
}

