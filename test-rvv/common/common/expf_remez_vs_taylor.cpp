/*
 * expf_remez_test.cpp — Compare Remez polynomial expf approximation with std::expf.
 *
 * Implements "reduction → approximation → reconstruction":
 *   reduction: x = n*ln2 + r,  n = round(x/ln2),  r in [-ln2/2, ln2/2]
 *   approximation: exp(r) ≈ P(r) (Remez degree 7)  or Taylor fallback (degree 7)
 *   reconstruction: exp(x) = 2^n * exp(r), where 2^n is loaded from a small table.
 *
 * Build:
 *   make expf_remez_test ARCH=riscv
 *
 * Run:
 *   make run_expf_remez_test ARCH=riscv
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#ifdef __RVV10__
#include <riscv_vector.h>
#endif

namespace {

// ---------------- Range reduction constants ----------------
// python3 parms.py 得到的参数
static constexpr float kLog2Inv = 1.442695040888963f;     // 1 / ln(2)
static constexpr float kLog2Hi = 0.6931471824645996f;     // ln(2) high
static constexpr float kLog2Lo = -1.904654290582768e-09f; // ln(2) low

static constexpr float kXMax = 88.0f; // exp(88) ~ 1.6e38, close to float max
static constexpr float kXMin =
    -88.0f; // exp(-88) ~ 6e-39, close to float subnormal range

// 2^-127 as a float (IEEE754 subnormal: exp=0, mantissa=2^22).
// python3 parms.py 得到的参数
static constexpr float kTwoToMinus127 = 5.877471754111438e-39f;

// ---------------- Remez coefficients for exp(r) on [0, ln(2)] (degree 7)
// python3 parm_remez_exp.py 得到的参数
static constexpr float expf_remez_c0 = 9.9999999998e-01f;
static constexpr float expf_remez_c1 = 1.0000000154e+00f;
static constexpr float expf_remez_c2 = 4.9999959620e-01f;
static constexpr float expf_remez_c3 = 1.6667078702e-01f;
static constexpr float expf_remez_c4 = 4.1645250213e-02f;
static constexpr float expf_remez_c5 = 8.3952782982e-03f;
static constexpr float expf_remez_c6 = 1.2887034349e-03f;
static constexpr float expf_remez_c7 = 2.8147688485e-04f;

// ---------------- Taylor coefficients (1/k!), k>=2 ----------------
static constexpr float taylor_c2 = 0.5f;
static constexpr float taylor_c3 = 1.0f / 6.0f;
static constexpr float taylor_c4 = 1.0f / 24.0f;
static constexpr float taylor_c5 = 1.0f / 120.0f;
static constexpr float taylor_c6 = 1.0f / 720.0f;
static constexpr float taylor_c7 = 1.0f / 5040.0f;
static constexpr float kMaxAbsRefThreshold = 1e10f;

// ---------------- Error metrics ----------------
struct ErrorStats {
  float max_abs_err;
  float max_rel_err;   // 最能体现误差大小的指标
  double mean_abs_err; // 由于 exp(x) 增长极快，大数值点的绝对误差会拉高平均值
};

struct WorstCasePoint {
  std::size_t idx_abs;
  std::size_t idx_rel;
};

// Restrict max absolute error to moderate-magnitude references (avoid exp(88)
// dominating).
static ErrorStats
compute_errors(const std::vector<float>& ref,
               const std::vector<float>& approx,
               float max_abs_ref = kMaxAbsRefThreshold)
{
  const std::size_t n = ref.size();
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  double sum_abs = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const float r = ref[i];
    const float a = approx[i];
    const float abs_err = std::fabs(a - r);
    sum_abs += static_cast<double>(abs_err);

    if (std::fabs(r) <= max_abs_ref) {
      if (abs_err > max_abs)
        max_abs = abs_err;
    }

    if (r != 0.0f) {
      const float rel = abs_err / std::fabs(r);
      if (rel > max_rel)
        max_rel = rel;
    }
  }

  ErrorStats s;
  s.max_abs_err = max_abs;
  s.max_rel_err = max_rel;
  s.mean_abs_err = sum_abs / static_cast<double>(n);
  return s;
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

// Create exact 2^n in float by building IEEE754 exponent/mantissa.
// This matches `std::ldexpf(1.0f, n)` for all representable float results,
// including subnormals.
static inline float
pow2f_from_int(int32_t n)
{
  using u32 = std::uint32_t;
  union {
    u32 u;
    float f;
  } pun{};

  // float exponent range: normal uses unbiased exponent [-126, 127]
  if (n > 127)
    return std::numeric_limits<float>::infinity();

  // Subnormal range: n in [-149, -127] => exponent bits = 0, mantissa = 2^(n+149)
  if (n < -149)
    return 0.0f;

  if (n >= -126) {
    const int exp_biased = n + 127; // 1..255 (255 => inf)
    pun.u = static_cast<u32>(exp_biased) << 23;
    return pun.f;
  }

  const int exp_sub = n + 149; // 0..22
  pun.u = static_cast<u32>(1u) << exp_sub;
  return pun.f;
}

static inline float
pow2f_from_int_simple(int32_t n)
{
  using u32 = std::uint32_t;
  union {
    u32 u;
    float f;
  } pun;

  // 1. 模拟向量中的 vadd.vx n_i, 127
  int32_t exp_offset = n + 127;

  // 2. 模拟向量中的 vmax.vx 和 vmin.vx (安全钳位)
  // 确保 exp_offset 落在 [0, 255] 范围内
  // 0   会构造出 0.0f
  // 255 会构造出 Infinity
  if (exp_offset <= 0) {
    pun.f = kTwoToMinus127;
    return pun.f;
  }
  if (exp_offset > 255)
    exp_offset = 255;

  // 3. 模拟向量中的 vsll.vx ..., 23
  // 构造 IEEE 754 位模式
  pun.u = static_cast<u32>(exp_offset) << 23;

  // 4. 模拟向量中的 vreinterpret
  return pun.f;
}

// ---------------- Scalar expf: Remez ----------------
static float
expf_remez_scalar(float x)
{
  if (std::isnan(x))
    return std::numeric_limits<float>::quiet_NaN();
  if (std::isinf(x))
    return (x > 0.f) ? std::numeric_limits<float>::infinity() : 0.f;

  x = std::fmax(std::fmin(x, kXMax), kXMin);

  // reduction
  float flt_n = x * kLog2Inv;
  auto n = static_cast<int32_t>(std::round(flt_n));
  flt_n = static_cast<float>(n);
  float r = x - flt_n * kLog2Hi - flt_n * kLog2Lo;

  // approximation: exp(r) ≈ P(r)
  float poly = expf_remez_c7;
  poly = expf_remez_c6 + r * poly;
  poly = expf_remez_c5 + r * poly;
  poly = expf_remez_c4 + r * poly;
  poly = expf_remez_c3 + r * poly;
  poly = expf_remez_c2 + r * poly;
  poly = expf_remez_c1 + r * poly;
  float exp_r = expf_remez_c0 + r * poly;

  // reconstruction: exp(x) = 2^n * exp(r)
  // return exp_r * pow2f_from_int(n);
  return exp_r * std::ldexpf(1.0f, n);
}

// ---------------- Scalar expf: Taylor fallback ----------------
static float
expf_taylor_scalar(float x)
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

  // exp(r) = 1 + r + r^2/2! + ... + r^7/7!
  // Implement as: exp(r) = 1 + r * (1 + (c2 + r*(c3 + ... + r*c7)))
  float poly = taylor_c7;
  poly = taylor_c6 + r * poly;
  poly = taylor_c5 + r * poly;
  poly = taylor_c4 + r * poly;
  poly = taylor_c3 + r * poly;
  poly = taylor_c2 + r * poly;

  float exp_r = 1.0f + r * (1.0f + poly);
  // return exp_r * pow2f_from_int(n);
  return exp_r * std::ldexpf(1.0f, n);
}

#ifdef __RVV10__
// ---------------- RVV expf: Remez ----------------
static float exp2_table[256];
static bool exp2_table_init = false;

static void
init_exp2_table()
{
  if (exp2_table_init)
    return;
  for (int i = 0; i < 256; ++i) {
    int n = i - 127;
    // exp2_table[i] = pow2f_from_int(n);
    exp2_table[i] = std::ldexpf(1.0f, n);
  }
  exp2_table_init = true;
}

static void
expf_RVV_remez_loop(const float* xs, float* out, std::size_t n)
{
  // No table lookup: build exact 2^n via IEEE754 exponent/mantissa bits.

  std::size_t j = 0;
  while (j < n) {
    const std::size_t vl = __riscv_vsetvl_e32m2(n - j);

    vfloat32m2_t vx = __riscv_vle32_v_f32m2(xs + j, vl);
    vx = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(vx, kXMin, vl), kXMax, vl);

    // n = round(x / ln2)
    vfloat32m2_t flt_n = __riscv_vfmul_vf_f32m2(vx, kLog2Inv, vl);
    vint32m2_t n_i = __riscv_vfcvt_x_f_v_i32m2(flt_n, vl);
    flt_n = __riscv_vfcvt_f_x_v_f32m2(n_i, vl);

    // r = x - n*ln2
    vfloat32m2_t r = __riscv_vfnmsub_vf_f32m2(flt_n, kLog2Hi, vx, vl);
    r = __riscv_vfnmsub_vf_f32m2(flt_n, kLog2Lo, r, vl);

    // poly = c7; poly = c6 + r*poly; ... ; exp_r = c0 + r*poly
    vfloat32m2_t poly = __riscv_vfmv_v_f_f32m2(expf_remez_c7, vl);
    poly =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c6, vl), r, poly,
        vl);
    poly =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c5, vl), r, poly,
        vl);
    poly =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c4, vl), r, poly,
        vl);
    poly =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c3, vl), r, poly,
        vl);
    poly =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c2, vl), r, poly,
        vl);
    poly =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c1, vl), r, poly,
        vl);
    vfloat32m2_t exp_r =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c0, vl), r, poly,
        vl);

    // Exact reconstruction for exp(x) = 2^n * exp(r)
    // exp(x) factor: two_n = 2^n.
    // Note: with x clamped to [-88, 88], the only subnormal case is n == -127
    // (i.e. exp_offset == 0). For that lane we must use 2^-127, not 0.
    vint32m2_t exp_offset = __riscv_vadd_vx_i32m2(n_i, 127, vl);
    exp_offset = __riscv_vmax_vx_i32m2(exp_offset, 0, vl);
    exp_offset = __riscv_vmin_vx_i32m2(exp_offset, 255, vl);

    vuint32m2_t res_bits =
        __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(exp_offset), 23, vl);
    vfloat32m2_t two_n_normal = __riscv_vreinterpret_v_u32m2_f32m2(res_bits);

    const vbool16_t is_n_neg127 = __riscv_vmseq_vx_i32m2_b16(n_i, -127, vl);
    vfloat32m2_t two_n_sub = __riscv_vfmv_v_f_f32m2(kTwoToMinus127, vl);
    vfloat32m2_t two_n =
        __riscv_vmerge_vvm_f32m2(two_n_normal, two_n_sub, is_n_neg127, vl);

    vfloat32m2_t result = __riscv_vfmul_vv_f32m2(exp_r, two_n, vl);
    __riscv_vse32_v_f32m2(out + j, result, vl);
    j += vl;
  }
}

// ---------------- RVV expf: Remez (table lookup 2^n) ----------------
static void
expf_RVV_remez_loop_table(const float* xs, float* out, std::size_t n)
{
  init_exp2_table();

  std::size_t j = 0;
  while (j < n) {
    const std::size_t vl = __riscv_vsetvl_e32m2(n - j);

    vfloat32m2_t vx = __riscv_vle32_v_f32m2(xs + j, vl);
    vx = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(vx, kXMin, vl), kXMax, vl);

    // n = round(x / ln2)
    vfloat32m2_t flt_n = __riscv_vfmul_vf_f32m2(vx, kLog2Inv, vl);
    vint32m2_t n_i = __riscv_vfcvt_x_f_v_i32m2(flt_n, vl);
    flt_n = __riscv_vfcvt_f_x_v_f32m2(n_i, vl);

    // r = x - n*ln2
    vfloat32m2_t r = __riscv_vfnmsub_vf_f32m2(flt_n, kLog2Hi, vx, vl);
    r = __riscv_vfnmsub_vf_f32m2(flt_n, kLog2Lo, r, vl);

    // poly = c7; poly = c6 + r*poly; ... ; exp_r = c0 + r*poly
    vfloat32m2_t poly = __riscv_vfmv_v_f_f32m2(expf_remez_c7, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c6, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c5, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c4, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c3, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c2, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c1, vl), r, poly, vl);
    vfloat32m2_t exp_r =
        __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(expf_remez_c0, vl), r, poly, vl);

    // table lookup for 2^n
    vint32m2_t idx_i = __riscv_vadd_vx_i32m2(n_i, 127, vl);
    idx_i = __riscv_vmax_vx_i32m2(idx_i, 0, vl);
    idx_i = __riscv_vmin_vx_i32m2(idx_i, 255, vl);
    vuint32m2_t idx_u = __riscv_vreinterpret_v_i32m2_u32m2(idx_i);
    vuint32m2_t idx_byte =
        __riscv_vmul_vx_u32m2(idx_u, static_cast<uint32_t>(sizeof(float)), vl);
    vfloat32m2_t two_n = __riscv_vluxei32_v_f32m2(exp2_table, idx_byte, vl);

    vfloat32m2_t result = __riscv_vfmul_vv_f32m2(exp_r, two_n, vl);
    __riscv_vse32_v_f32m2(out + j, result, vl);
    j += vl;
  }
}

// ---------------- RVV expf: Taylor ----------------
static void
expf_RVV_taylor_loop(const float* xs, float* out, std::size_t n)
{
  init_exp2_table();

  std::size_t j = 0;
  while (j < n) {
    const std::size_t vl = __riscv_vsetvl_e32m2(n - j);

    vfloat32m2_t vx = __riscv_vle32_v_f32m2(xs + j, vl);
    vx = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(vx, kXMin, vl), kXMax, vl);

    vfloat32m2_t flt_n = __riscv_vfmul_vf_f32m2(vx, kLog2Inv, vl);
    vint32m2_t n_i = __riscv_vfcvt_x_f_v_i32m2(flt_n, vl);
    flt_n = __riscv_vfcvt_f_x_v_f32m2(n_i, vl);

    vfloat32m2_t r = __riscv_vfnmsub_vf_f32m2(flt_n, kLog2Hi, vx, vl);
    r = __riscv_vfnmsub_vf_f32m2(flt_n, kLog2Lo, r, vl);

    // poly = c7; ... ; poly = c2 + r*... ; then exp_r = 1 + r*(1 + poly)
    vfloat32m2_t poly = __riscv_vfmv_v_f_f32m2(taylor_c7, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(taylor_c6, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(taylor_c5, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(taylor_c4, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(taylor_c3, vl), r, poly, vl);
    poly = __riscv_vfmacc_vv_f32m2(__riscv_vfmv_v_f_f32m2(taylor_c2, vl), r, poly, vl);

    vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t one_plus_poly = __riscv_vfadd_vv_f32m2(one, poly, vl);
    vfloat32m2_t exp_r =
        __riscv_vfmacc_vv_f32m2(one, r, one_plus_poly, vl); // one + r*(one+poly)

    // 使用查表方式计算 2^n
    vint32m2_t idx_i = __riscv_vadd_vx_i32m2(n_i, 127, vl);
    idx_i = __riscv_vmax_vx_i32m2(idx_i, 0, vl);
    idx_i = __riscv_vmin_vx_i32m2(idx_i, 255, vl);

    vuint32m2_t idx_u = __riscv_vreinterpret_v_i32m2_u32m2(idx_i);
    vuint32m2_t idx_byte =
        __riscv_vmul_vx_u32m2(idx_u, static_cast<uint32_t>(sizeof(float)), vl);
    vfloat32m2_t two_n = __riscv_vluxei32_v_f32m2(exp2_table, idx_byte, vl);

    vfloat32m2_t result = __riscv_vfmul_vv_f32m2(exp_r, two_n, vl);
    __riscv_vse32_v_f32m2(out + j, result, vl);
    j += vl;
  }
}
#endif // __RVV10__

} // namespace

int
main()
{
  const std::size_t n = 10000;
  const int iters = 100;

  std::vector<float> xs(n);
  std::vector<float> ref(n);

  for (std::size_t i = 0; i < n; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(n - 1);
    const float x = kXMin + (kXMax - kXMin) * t;
    xs[i] = x;
    ref[i] = std::expf(x);
  }

  std::vector<float> out_scalar_remez(n);
  std::vector<float> out_scalar_taylor(n);

  for (std::size_t i = 0; i < n; ++i) {
    out_scalar_remez[i] = expf_remez_scalar(xs[i]);
    out_scalar_taylor[i] = expf_taylor_scalar(xs[i]);
  }

  auto s_ref = compute_errors(ref, out_scalar_remez);
  auto s_tay = compute_errors(ref, out_scalar_taylor);

  std::printf("=== Error metric notes ===\n");
  // std::printf("    - max absolute error: max_i |approx_i - ref_i|, only for |ref_i| <= %.3e\n",
  //             kMaxAbsRefThreshold);
  std::printf("    - max relative error: max_i (|approx_i - ref_i| / |ref_i|), ref_i != 0\n");
  // std::printf("    - mean absolute error: (1/N) * sum_i |approx_i - ref_i|\n");
  // std::printf("    - max_abs threshold:   |ref| <= %.3e\n", kMaxAbsRefThreshold);
  std::printf("\n");

  std::printf("=== expf approximation vs std::expf (n = %zu) ===\n", n);
  std::printf("  Scalar Remez (degree 7):\n");
  // std::printf("    max absolute error:  %.6e\n", s_ref.max_abs_err);
  std::printf("    max relative error:  %.6e\n", s_ref.max_rel_err);
  // std::printf("    mean absolute error: %.6e\n", s_ref.mean_abs_err);
  // print_worst_case_details("Scalar Remez", xs, ref, out_scalar_remez, kMaxAbsRefThreshold);

  std::printf("  Scalar Taylor (degree 7):\n");
  // std::printf("    max absolute error:  %.6e\n", s_tay.max_abs_err);
  std::printf("    max relative error:  %.6e\n", s_tay.max_rel_err);
  // std::printf("    mean absolute error: %.6e\n", s_tay.mean_abs_err);
  // print_worst_case_details("Scalar Taylor", xs, ref, out_scalar_taylor);

#ifdef __RVV10__
  std::vector<float> out_rvv_remez(n);
  std::vector<float> out_rvv_remez_table(n);
  std::vector<float> out_rvv_taylor(n);

  expf_RVV_remez_loop_table(xs.data(), out_rvv_remez_table.data(), n);
  expf_RVV_remez_loop(xs.data(), out_rvv_remez.data(), n);
  expf_RVV_taylor_loop(xs.data(), out_rvv_taylor.data(), n);

  auto s_rvv_remez_table = compute_errors(ref, out_rvv_remez_table);
  auto s_rvv_remez = compute_errors(ref, out_rvv_remez);
  auto s_rvv_tay = compute_errors(ref, out_rvv_taylor);

  std::printf("  RVV Remez (table lookup 2^n):\n");
  // std::printf("    max absolute error:  %.6e\n", s_rvv_remez_table.max_abs_err);
  std::printf("    max relative error:  %.6e\n", s_rvv_remez_table.max_rel_err);
  // print_worst_case_details("RVV Remez [table 2^n]", xs, ref, out_rvv_remez_table);

  std::printf("  RVV Remez (construct 2^n):\n");
  // std::printf("    max absolute error:  %.6e\n", s_rvv_remez.max_abs_err);
  std::printf("    max relative error:  %.6e\n", s_rvv_remez.max_rel_err);
  // std::printf("    mean absolute error: %.6e\n", s_rvv_remez.mean_abs_err);
  // print_worst_case_details("RVV Remez [construct 2^n]", xs, ref, out_rvv_remez, kMaxAbsRefThreshold);

  std::printf("  RVV Taylor:\n");
  // std::printf("    max absolute error:  %.6e\n", s_rvv_tay.max_abs_err);
  std::printf("    max relative error:  %.6e\n", s_rvv_tay.max_rel_err);
  // std::printf("    mean absolute error: %.6e\n", s_rvv_tay.mean_abs_err);
  // print_worst_case_details("RVV Taylor", xs, ref, out_rvv_taylor);

  float max_diff_remez_table = 0.0f;
  float max_diff_remez = 0.0f;
  float max_diff_tay = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float d0 = std::fabs(out_rvv_remez_table[i] - out_scalar_remez[i]);
    if (d0 > max_diff_remez_table)
      max_diff_remez_table = d0;
    const float d1 = std::fabs(out_rvv_remez[i] - out_scalar_remez[i]);
    if (d1 > max_diff_remez)
      max_diff_remez = d1;
    const float d2 = std::fabs(out_rvv_taylor[i] - out_scalar_taylor[i]);
    if (d2 > max_diff_tay)
      max_diff_tay = d2;
  }

  std::printf("  RVV Remez [table 2^n] vs scalar Remez max diff: %.6e\n", max_diff_remez_table);
  std::printf("  RVV Remez [construct 2^n] vs scalar Remez max diff: %.6e\n", max_diff_remez);
  std::printf("  RVV Taylor vs scalar Taylor max diff: %.6e\n", max_diff_tay);
#else
  std::printf("  (RVV path disabled: compile without __RVV10__)\n");
#endif

  // ---------------- Performance ----------------
  std::vector<float> tmp(n);

  auto time_ms = [](auto&& fn) -> double {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  double t_std = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      for (std::size_t i = 0; i < n; ++i)
        tmp[i] = std::expf(xs[i]);
  });

  double t_scalar_remez = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      for (std::size_t i = 0; i < n; ++i)
        tmp[i] = expf_remez_scalar(xs[i]);
  });

  double t_scalar_taylor = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      for (std::size_t i = 0; i < n; ++i)
        tmp[i] = expf_taylor_scalar(xs[i]);
  });

  std::printf("\n=== Performance (n = %zu, %d iters) ===\n", n, iters);
  std::printf("  std::expf:        %8.3f ms\n", t_std);
  //  std::printf("  Scalar Remez:    %8.3f ms  (%.2fx vs std)\n", t_scalar_remez,
  //  t_scalar_remez / t_std);
  std::printf("  Scalar Remez:    %8.3f ms  (speedup: %.2fx vs std)\n",
              t_scalar_remez,
              t_std / t_scalar_remez);
  //  std::printf("  Scalar Taylor:   %8.3f ms  (%.2fx vs std)\n", t_scalar_taylor,
  //  t_scalar_taylor / t_std);
  std::printf("  Scalar Taylor:   %8.3f ms  (speedup: %.2fx vs std)\n",
              t_scalar_taylor,
              t_std / t_scalar_taylor);

#ifdef __RVV10__
  double t_rvv_remez_table = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      expf_RVV_remez_loop_table(xs.data(), tmp.data(), n);
  });

  double t_rvv_remez = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      expf_RVV_remez_loop(xs.data(), tmp.data(), n);
  });

  double t_rvv_taylor = time_ms([&]() {
    for (int k = 0; k < iters; ++k)
      expf_RVV_taylor_loop(xs.data(), tmp.data(), n);
  });

  //  std::printf("  RVV Remez:       %8.3f ms  (%.2fx vs std, %.2fx vs scalar)\n",
  // t_rvv_remez, t_rvv_remez / t_std, t_rvv_remez / t_scalar_remez);
  std::printf("  RVV Remez [table 2^n]:      %8.3f ms  (speedup: %.2fx vs std, %.2fx vs scalar)\n",
              t_rvv_remez_table,
              t_std / t_rvv_remez_table,
              t_scalar_remez / t_rvv_remez_table);
  std::printf("  RVV Remez [construct 2^n]:  %8.3f ms  (speedup: %.2fx vs std, %.2fx vs scalar)\n",
              t_rvv_remez,
              t_std / t_rvv_remez,
              t_scalar_remez / t_rvv_remez);
  //  std::printf("  RVV Taylor:      %8.3f ms  (%.2fx vs std, %.2fx vs scalar)\n",
  // t_rvv_taylor, t_rvv_taylor / t_std, t_rvv_taylor / t_scalar_taylor);
  std::printf("  RVV Taylor:      %8.3f ms  (speedup: %.2fx vs std, %.2fx vs scalar)\n",
              t_rvv_taylor,
              t_std / t_rvv_taylor,
              t_scalar_taylor / t_rvv_taylor);
#endif

  return 0;
}
