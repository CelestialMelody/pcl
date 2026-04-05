/*
 * atan2_test.cpp — Compare atan2 approximation (Hastings-style polynomial)
 * with std::atan2. Uses the same coefficients as pcl::atan2_RVV_f32m2.
 *
 * Build (standalone, scalar only):
 *   g++ -std=c++17 -O3 -o atan2_test atan2_test.cpp -lm
 *
 * Build (RVV path, with PCL common):
 *   make atan2_test ARCH=riscv
 *
 * Run:
 *   ./atan2_test
 *   make run_atan2_test ARCH=riscv   # under QEMU when cross-building
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <chrono>

#if defined(__RVV10__)
#include <riscv_vector.h>
#include <pcl/common/common.h>
#include <pcl/common/impl/common.hpp>
#include <cstddef>
#endif

namespace {

// Same coefficients as pcl::atan2_RVV_f32m2 (Hastings-style, ~0.001 deg max error)
// parms form https://mazzo.li/posts/vectorized-atan2.html
const float a1 = 0.99997726f;
const float a3 = -0.33262347f;
const float a5 = 0.19354346f;
const float a7 = -0.11643287f;
const float a9 = 0.05265332f;
const float a11 = -0.01172120f;
const float pi = 3.14159265358979323846f;
const float pi_2 = 1.57079632679489661923f;


// 使用 python parm_remez_atan2.py 得到的参数

// === atan2 approximation vs std::atan2 (n = 65536) ===
//   Scalar approximation (same polynomial as RVV):
//     max absolute error:  0.000232 rad  (0.0133 deg)
//     mean absolute error: 0.000089 rad
//   RVV (pcl::atan2_RVV_f32m2):
//     max absolute error:  0.000002 rad  (0.0001 deg) --> 使用 https://mazzo.li/posts/vectorized-atan2.html 的参数精度更高
//     mean absolute error: 0.000001 rad
//   RVV vs scalar approx max diff: 2.326965e-04 (expect ~0)


// const float a1 = 0.9985091188034383f;
// const float a3 = -0.3223280401559811f;
// const float a5 = 0.1642503213121442f;
// const float a7 = -0.07457206137001779f;
// const float a9 = 0.02284963045699147f;
// const float a11 = -0.003310805664035673f;

// const float pi = 3.141592653589793115998f;
// const float pi_2 = 1.570796326794896557999f;

const float tiny_f = 1e-20f;

// Scalar atan2 approximation (same algorithm as RVV version)
float atan2_approx(float y, float x)
{
  float abs_x = std::fabs(x);
  float abs_y = std::fabs(y);
  bool swap = abs_x < abs_y;
  float num = swap ? x : y;
  float den = swap ? y : x;
  if (std::fabs(den) < tiny_f)
    den = (den >= 0.f) ? tiny_f : -tiny_f;
  float atan_input = num / den;

  float x2 = atan_input * atan_input;
  float p = a11;
  p = a9 + x2 * p;
  p = a7 + x2 * p;
  p = a5 + x2 * p;
  p = a3 + x2 * p;
  p = a1 + x2 * p;
  float result = atan_input * p;

  if (swap)
    result = (atan_input >= 0.f ? pi_2 : -pi_2) - result;
  if (x < 0.f)
    result += (y >= 0.f ? pi : -pi);

  return result;
}

void run_scalar_vs_std(const float* ys, const float* xs, float* out_approx, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i)
    out_approx[i] = atan2_approx(ys[i], xs[i]);
}

void run_std_atan2(const float* ys, const float* xs, float* out_std, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i)
    out_std[i] = std::atan2(ys[i], xs[i]);
}

#if defined(__RVV10__)
void run_rvv_vs_std(const float* ys, const float* xs, float* out_rvv, std::size_t n)
{
  std::size_t j = 0;
  while (j < n) {
    std::size_t vl = __riscv_vsetvl_e32m2(n - j);
    vfloat32m2_t v_y = __riscv_vle32_v_f32m2(ys + j, vl);
    vfloat32m2_t v_x = __riscv_vle32_v_f32m2(xs + j, vl);
    vfloat32m2_t v_out = pcl::atan2_RVV_f32m2(v_y, v_x, vl);
    __riscv_vse32_v_f32m2(out_rvv + j, v_out, vl);
    j += vl;
  }
}
#endif

void compute_errors(const float* ref, const float* approx, std::size_t n,
                    float& max_abs_rad, float& max_abs_deg, double& mean_abs_rad)
{
  max_abs_rad = 0.f;
  mean_abs_rad = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    float e = std::fabs(approx[i] - ref[i]);
    if (e > max_abs_rad)
      max_abs_rad = e;
    mean_abs_rad += static_cast<double>(e);
  }
  mean_abs_rad /= static_cast<double>(n);
  const float rad2deg = 180.f / pi;
  max_abs_deg = max_abs_rad * rad2deg;
}

} // namespace

int main()
{
  const std::size_t n = 256 * 256;
  float* ys = static_cast<float*>(std::malloc(n * sizeof(float)));
  float* xs = static_cast<float*>(std::malloc(n * sizeof(float)));
  float* ref = static_cast<float*>(std::malloc(n * sizeof(float)));
  float* out_approx = static_cast<float*>(std::malloc(n * sizeof(float)));
#if defined(__RVV10__)
  float* out_rvv = static_cast<float*>(std::malloc(n * sizeof(float)));
#endif

  if (!ys || !xs || !ref || !out_approx) {
    std::fprintf(stderr, "malloc failed\n");
    std::free(ys);
    std::free(xs);
    std::free(ref);
    std::free(out_approx);
#if defined(__RVV10__)
    std::free(out_rvv);
#endif
    return 1;
  }

  // Test grid: [-1,1] x [-1,1] plus edge cases
  std::size_t idx = 0;
  for (int iy = 0; iy < 256; ++iy) {
    float y = (iy == 0) ? -1.f : ((iy == 255) ? 1.f : (-1.f + 2.f * iy / 255.f));
    for (int ix = 0; ix < 256; ++ix) {
      float x = (ix == 0) ? -1.f : ((ix == 255) ? 1.f : (-1.f + 2.f * ix / 255.f));
      ys[idx] = y;
      xs[idx] = x;
      ref[idx] = std::atan2(y, x);
      ++idx;
    }
  }

  run_scalar_vs_std(ys, xs, out_approx, n);
  float max_abs_rad_s, max_abs_deg_s;
  double mean_abs_rad_s;
  compute_errors(ref, out_approx, n, max_abs_rad_s, max_abs_deg_s, mean_abs_rad_s);

  std::printf("=== atan2 approximation vs std::atan2 (n = %zu) ===\n", n);
  std::printf("  Scalar approximation (same polynomial as RVV):\n");
  std::printf("    max absolute error:  %.6f rad  (%.4f deg)\n", max_abs_rad_s, max_abs_deg_s);
  std::printf("    mean absolute error: %.6f rad\n", mean_abs_rad_s);

#if defined(__RVV10__)
  run_rvv_vs_std(ys, xs, out_rvv, n);
  float max_abs_rad_r, max_abs_deg_r;
  double mean_abs_rad_r;
  compute_errors(ref, out_rvv, n, max_abs_rad_r, max_abs_deg_r, mean_abs_rad_r);
  std::printf("  RVV (pcl::atan2_RVV_f32m2):\n");
  std::printf("    max absolute error:  %.6f rad  (%.4f deg)\n", max_abs_rad_r, max_abs_deg_r);
  std::printf("    mean absolute error: %.6f rad\n", mean_abs_rad_r);

  // RVV vs scalar approx (should match within float rounding)
  float max_diff = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    float d = std::fabs(out_rvv[i] - out_approx[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::printf("  RVV vs scalar approx max diff: %.6e (expect ~0)\n", max_diff);
#endif

  // Performance benchmark
  std::printf("\n=== Performance Benchmark (n = %zu, iterations = 100) ===\n", n);
  const int iterations = 100;
  float* out_std = static_cast<float*>(std::malloc(n * sizeof(float)));
  if (!out_std) {
    std::fprintf(stderr, "malloc failed for out_std\n");
    std::free(ys);
    std::free(xs);
    std::free(ref);
    std::free(out_approx);
#if defined(__RVV10__)
    std::free(out_rvv);
#endif
    return 1;
  }

  // Warmup
  run_std_atan2(ys, xs, out_std, n);
  run_scalar_vs_std(ys, xs, out_approx, n);
#if defined(__RVV10__)
  run_rvv_vs_std(ys, xs, out_rvv, n);
#endif

  // Benchmark std::atan2
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    run_std_atan2(ys, xs, out_std, n);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_std = std::chrono::duration<double, std::milli>(end - start).count();
  double throughput_std = (static_cast<double>(n) * iterations) / (time_std / 1000.0) / 1e6; // M elements/sec

  // Benchmark scalar approximation
  start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    run_scalar_vs_std(ys, xs, out_approx, n);
  }
  end = std::chrono::high_resolution_clock::now();
  double time_scalar = std::chrono::duration<double, std::milli>(end - start).count();
  double throughput_scalar = (static_cast<double>(n) * iterations) / (time_scalar / 1000.0) / 1e6;

  std::printf("  std::atan2:\n");
  std::printf("    time:       %.3f ms (%.3f ms/iter)\n", time_std, time_std / iterations);
  std::printf("    throughput: %.2f M elements/sec\n", throughput_std);
  std::printf("  Scalar approximation:\n");
  std::printf("    time:       %.3f ms (%.3f ms/iter)\n", time_scalar, time_scalar / iterations);
  std::printf("    throughput: %.2f M elements/sec\n", throughput_scalar);
  std::printf("    speedup:    %.2fx vs std::atan2\n", time_std / time_scalar);

#if defined(__RVV10__)
  // Benchmark RVV
  start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    run_rvv_vs_std(ys, xs, out_rvv, n);
  }
  end = std::chrono::high_resolution_clock::now();
  double time_rvv = std::chrono::duration<double, std::milli>(end - start).count();
  double throughput_rvv = (static_cast<double>(n) * iterations) / (time_rvv / 1000.0) / 1e6;

  std::printf("  RVV (pcl::atan2_RVV_f32m2):\n");
  std::printf("    time:       %.3f ms (%.3f ms/iter)\n", time_rvv, time_rvv / iterations);
  std::printf("    throughput: %.2f M elements/sec\n", throughput_rvv);
  std::printf("    speedup:    %.2fx vs std::atan2, %.2fx vs scalar\n",
              time_std / time_rvv, time_scalar / time_rvv);
#endif

  std::free(out_std);
#if defined(__RVV10__)
  std::free(out_rvv);
#endif

  std::free(ys);
  std::free(xs);
  std::free(ref);
  std::free(out_approx);
  return 0;
}
