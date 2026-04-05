/*
 * expf_test.cpp — Compare scalar/RVV expf approximation with std::expf
 *
 * RVV path directly calls pcl::expf_RVV_f32m2 from common.hpp.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#if defined(__RVV10__)
#include <cstddef>
#include <pcl/common/common.h>
#include <pcl/common/impl/common.hpp>
#include <riscv_vector.h>
#endif

namespace {

static constexpr float kLog2Inv = 1.442695040888963f;
static constexpr float kLog2Hi = 0.6931471824645996f;
static constexpr float kLog2Lo = -1.904654290582768e-09f;
static constexpr float kXMax = 88.0f;
static constexpr float kXMin = -88.0f;

// Same Remez coefficients as expf_remez_vs_taylor.cpp
static constexpr float expf_remez_c0 = 9.9999999998e-01f;
static constexpr float expf_remez_c1 = 1.0000000154e+00f;
static constexpr float expf_remez_c2 = 4.9999959620e-01f;
static constexpr float expf_remez_c3 = 1.6667078702e-01f;
static constexpr float expf_remez_c4 = 4.1645250213e-02f;
static constexpr float expf_remez_c5 = 8.3952782982e-03f;
static constexpr float expf_remez_c6 = 1.2887034349e-03f;
static constexpr float expf_remez_c7 = 2.8147688485e-04f;
static constexpr float kMaxAbsRefThreshold = 1e10f;

struct WorstCasePoint {
  std::size_t idx_abs;
  std::size_t idx_rel;
};

static float
expf_remez_scalar(float x)
{
  if (std::isnan(x))
    return std::numeric_limits<float>::quiet_NaN();
  if (std::isinf(x))
    return (x > 0.f) ? std::numeric_limits<float>::infinity() : 0.f;

  x = std::fmax(std::fmin(x, kXMax), kXMin);

  float flt_n = x * kLog2Inv;
  auto n = static_cast<int32_t>(std::round(flt_n));
  flt_n = static_cast<float>(n);
  float r = x - flt_n * kLog2Hi - flt_n * kLog2Lo;

  float poly = expf_remez_c7;
  poly = expf_remez_c6 + r * poly;
  poly = expf_remez_c5 + r * poly;
  poly = expf_remez_c4 + r * poly;
  poly = expf_remez_c3 + r * poly;
  poly = expf_remez_c2 + r * poly;
  poly = expf_remez_c1 + r * poly;
  float exp_r = expf_remez_c0 + r * poly;

  return exp_r * std::ldexpf(1.0f, n);
}

static void
run_std_expf(const std::vector<float>& xs, std::vector<float>& out_std)
{
  const std::size_t n = xs.size();
  for (std::size_t i = 0; i < n; ++i)
    out_std[i] = std::expf(xs[i]);
}

static void
run_scalar_remez(const std::vector<float>& xs, std::vector<float>& out)
{
  const std::size_t n = xs.size();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = expf_remez_scalar(xs[i]);
}

#if defined(__RVV10__)
static void
run_rvv_common(const std::vector<float>& xs, std::vector<float>& out)
{
  const std::size_t n = xs.size();
  std::size_t j = 0;
  while (j < n) {
    const std::size_t vl = __riscv_vsetvl_e32m2(n - j);
    const vfloat32m2_t vx = __riscv_vle32_v_f32m2(xs.data() + j, vl);
    const vfloat32m2_t vy = pcl::expf_RVV_f32m2(vx, vl);
    __riscv_vse32_v_f32m2(out.data() + j, vy, vl);
    j += vl;
  }
}
#endif

static void
compute_errors(const std::vector<float>& ref,
               const std::vector<float>& approx,
               float& max_abs_err,
               float& max_rel_err,
               double& mean_abs_err,
               float max_abs_ref = kMaxAbsRefThreshold)
{
  const std::size_t n = ref.size();
  max_abs_err = 0.0f;
  max_rel_err = 0.0f;
  mean_abs_err = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const float r = ref[i];
    const float a = approx[i];

    if (std::isnan(r) || std::isnan(a) || std::isinf(r) || std::isinf(a))
      continue;

    const float abs_err = std::fabs(a - r);
    if (std::fabs(r) <= max_abs_ref && abs_err > max_abs_err)
      max_abs_err = abs_err;
    mean_abs_err += static_cast<double>(abs_err);

    if (r != 0.0f) {
      const float rel = abs_err / std::fabs(r);
      if (rel > max_rel_err)
        max_rel_err = rel;
    }
  }

  mean_abs_err /= static_cast<double>(n);
}

static WorstCasePoint
find_worst_case_points(const std::vector<float>& ref,
                       const std::vector<float>& approx,
                       float max_abs_ref = kMaxAbsRefThreshold)
{
  const std::size_t n = ref.size();
  std::size_t idx_abs = 0;
  std::size_t idx_rel = 0;
  float best_abs = -1.0f;
  float best_rel = -1.0f;

  for (std::size_t i = 0; i < n; ++i) {
    const float r = ref[i];
    const float a = approx[i];
    if (std::isnan(r) || std::isnan(a) || std::isinf(r) || std::isinf(a))
      continue;

    const float abs_err = std::fabs(a - r);
    if (std::fabs(r) <= max_abs_ref && abs_err > best_abs) {
      best_abs = abs_err;
      idx_abs = i;
    }

    if (r != 0.0f) {
      const float rel = abs_err / std::fabs(r);
      if (rel > best_rel) {
        best_rel = rel;
        idx_rel = i;
      }
    }
  }

  WorstCasePoint p{};
  p.idx_abs = idx_abs;
  p.idx_rel = idx_rel;
  return p;
}

static void
reduce_x_to_n_r(float x, int32_t& n, float& r)
{
  float x_clamped = std::fmax(std::fmin(x, kXMax), kXMin);
  float flt_n = x_clamped * kLog2Inv;
  n = static_cast<int32_t>(std::round(flt_n));
  const float fn = static_cast<float>(n);
  r = x_clamped - fn * kLog2Hi - fn * kLog2Lo;
}

static void
print_worst_case_details(const char* label,
                         const std::vector<float>& xs,
                         const std::vector<float>& ref,
                         const std::vector<float>& approx,
                         float max_abs_ref = kMaxAbsRefThreshold)
{
  const auto w = find_worst_case_points(ref, approx, max_abs_ref);
  const std::size_t ia = w.idx_abs;
  const std::size_t ir = w.idx_rel;

  const float abs_err = std::fabs(approx[ia] - ref[ia]);
  const float rel_err =
      (ref[ir] != 0.0f) ? std::fabs(approx[ir] - ref[ir]) / std::fabs(ref[ir]) : 0.0f;

  int32_t n_abs = 0, n_rel = 0;
  float r_abs = 0.0f, r_rel = 0.0f;
  reduce_x_to_n_r(xs[ia], n_abs, r_abs);
  reduce_x_to_n_r(xs[ir], n_rel, r_rel);

  std::printf("    [%s] worst max_abs point:\n", label);
  std::printf("      idx=%zu, x=%.9g, ref=%.9g, approx=%.9g, abs_err=%.9g, n=%d, r=%.9g\n",
              ia, xs[ia], ref[ia], approx[ia], abs_err, n_abs, r_abs);
  std::printf("    [%s] worst max_rel point:\n", label);
  std::printf("      idx=%zu, x=%.9g, ref=%.9g, approx=%.9g, rel_err=%.9g, n=%d, r=%.9g\n",
              ir, xs[ir], ref[ir], approx[ir], rel_err, n_rel, r_rel);
}

} // namespace

int
main()
{
  const std::size_t n = 10000;
  const int iters = 100;

  std::vector<float> xs(n);
  std::vector<float> ref(n);
  std::vector<float> out_scalar(n);
  std::vector<float> out_std(n);
#if defined(__RVV10__)
  std::vector<float> out_rvv(n);
#endif

  for (std::size_t i = 0; i < n; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(n - 1);
    const float x = kXMin + (kXMax - kXMin) * t;
    xs[i] = x;
    ref[i] = std::expf(x);
  }

  run_scalar_remez(xs, out_scalar);

  float max_abs_s = 0.0f, max_rel_s = 0.0f;
  double mean_abs_s = 0.0;
  compute_errors(ref, out_scalar, max_abs_s, max_rel_s, mean_abs_s);


  std::printf("=== Error metric notes ===\n");
  // std::printf("    - max absolute error: max_i |approx_i - ref_i|, only for |ref_i| <= %.3e\n",
  //             kMaxAbsRefThreshold);
  std::printf("    - max relative error: max_i (|approx_i - ref_i| / |ref_i|), ref_i != 0\n");
  // std::printf("    - mean absolute error: (1/N) * sum_i |approx_i - ref_i|\n");
  // std::printf("    - max_abs threshold:   |ref| <= %.3e\n", kMaxAbsRefThreshold);
  std::printf("\n");

  std::printf("=== expf approximation vs std::expf (n = %zu) ===\n", n);
  std::printf("  Scalar Remez (aligned with common.hpp constants):\n");
  // std::printf("    max absolute error:  %.6e\n", max_abs_s);
  std::printf("    max relative error:  %.6e\n", max_rel_s);
  // std::printf("    mean absolute error: %.6e\n", mean_abs_s);
  // print_worst_case_details("Scalar Remez", xs, ref, out_scalar, kMaxAbsRefThreshold);

#if defined(__RVV10__)
  run_rvv_common(xs, out_rvv);
  float max_abs_r = 0.0f, max_rel_r = 0.0f;
  double mean_abs_r = 0.0;
  compute_errors(ref, out_rvv, max_abs_r, max_rel_r, mean_abs_r);
  std::printf("  RVV (pcl::expf_RVV_f32m2 from common.hpp):\n");
  // std::printf("    max absolute error:  %.6e\n", max_abs_r);
  std::printf("    max relative error:  %.6e\n", max_rel_r);
  // std::printf("    mean absolute error: %.6e\n", mean_abs_r);
  // print_worst_case_details("RVV common.hpp", xs, ref, out_rvv, kMaxAbsRefThreshold);

  float max_diff = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float d = std::fabs(out_rvv[i] - out_scalar[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::printf("  RVV vs scalar Remez max diff: %.6e\n", max_diff);
#endif

  auto time_ms = [](auto&& fn) -> double {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  // warmup
  run_std_expf(xs, out_std);
  run_scalar_remez(xs, out_scalar);
#if defined(__RVV10__)
  run_rvv_common(xs, out_rvv);
#endif

  const double t_std = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      run_std_expf(xs, out_std);
  });
  const double t_scalar = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      run_scalar_remez(xs, out_scalar);
  });

  std::printf("\n=== Performance (n = %zu, %d iters) ===\n", n, iters);
  std::printf("  std::expf:     %8.3f ms\n", t_std);
  std::printf("  Scalar Remez:  %8.3f ms  (speedup: %.2fx vs std)\n", t_scalar, t_std / t_scalar);

#if defined(__RVV10__)
  const double t_rvv = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      run_rvv_common(xs, out_rvv);
  });
  std::printf("  RVV common.hpp:%8.3f ms  (speedup: %.2fx vs std, %.2fx vs scalar)\n",
              t_rvv, t_std / t_rvv, t_scalar / t_rvv);
#endif

  return 0;
}
