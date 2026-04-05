/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_COMMON_IMPL_H_
#define PCL_COMMON_IMPL_H_

#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <cmath>
#include <cstdint>
#include <limits>

//////////////////////////////////////////////////////////////////////////////////////////////
inline double
pcl::getAngle3D (const Eigen::Vector4f &v1, const Eigen::Vector4f &v2, const bool in_degree)
{
  // Compute the actual angle
  double rad = v1.normalized ().dot (v2.normalized ());
  if (rad < -1.0)
    rad = -1.0;
  else if (rad >  1.0)
    rad = 1.0;
  return (in_degree ? std::acos (rad) * 180.0 / M_PI : std::acos (rad));
}

inline double
pcl::getAngle3D (const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const bool in_degree)
{
  // Compute the actual angle
  double rad = v1.normalized ().dot (v2.normalized ());
  if (rad < -1.0)
    rad = -1.0;
  else if (rad >  1.0)
    rad = 1.0;
  return (in_degree ? std::acos (rad) * 180.0 / M_PI : std::acos (rad));
}

#ifdef __SSE__
inline __m128
pcl::acos_SSE (const __m128 &x)
{
  /*
  This python code generates the coefficients:
  import math, numpy, scipy.optimize
  def get_error(S):
      err_sum=0.0
      for x in numpy.arange(0.0, 1.0, 0.0025):
          if (S[3]+S[4]*x)<0.0:
              err_sum+=10.0
          else:
              err_sum+=((S[0]+x*(S[1]+x*S[2]))*numpy.sqrt(S[3]+S[4]*x)+S[5]+x*(S[6]+x*S[7])-math.acos(x))**2.0
      return err_sum/400.0

  print(scipy.optimize.minimize(fun=get_error, x0=[1.57, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], method='Nelder-Mead', options={'maxiter':42000, 'maxfev':42000, 'disp':True, 'xatol':1e-6, 'fatol':1e-6}))
  */
  const __m128 mul_term = _mm_add_ps (_mm_set1_ps (1.59121552f), _mm_mul_ps (x, _mm_add_ps (_mm_set1_ps (-0.15461442f), _mm_mul_ps (x, _mm_set1_ps (0.05354897f)))));
  const __m128 add_term = _mm_add_ps (_mm_set1_ps (0.06681017f), _mm_mul_ps (x, _mm_add_ps (_mm_set1_ps (-0.09402311f), _mm_mul_ps (x, _mm_set1_ps (0.02708663f)))));
  return _mm_add_ps (_mm_mul_ps (mul_term, _mm_sqrt_ps (_mm_add_ps (_mm_set1_ps (0.89286965f), _mm_mul_ps (_mm_set1_ps (-0.89282669f), x)))), add_term);
}

inline __m128
pcl::getAcuteAngle3DSSE (const __m128 &x1, const __m128 &y1, const __m128 &z1, const __m128 &x2, const __m128 &y2, const __m128 &z2)
{
  const __m128 dot_product = _mm_add_ps (_mm_add_ps (_mm_mul_ps (x1, x2), _mm_mul_ps (y1, y2)), _mm_mul_ps (z1, z2));
  // The andnot-function realizes an abs-operation: the sign bit is removed
  // -0.0f (negative zero) means that all bits are 0, only the sign bit is 1
  return acos_SSE (_mm_min_ps (_mm_set1_ps (1.0f), _mm_andnot_ps (_mm_set1_ps (-0.0f), dot_product)));
}
#endif // ifdef __SSE__

#ifdef __AVX__
inline __m256
pcl::acos_AVX (const __m256 &x)
{
  const __m256 mul_term = _mm256_add_ps (_mm256_set1_ps (1.59121552f), _mm256_mul_ps (x, _mm256_add_ps (_mm256_set1_ps (-0.15461442f), _mm256_mul_ps (x, _mm256_set1_ps (0.05354897f)))));
  const __m256 add_term = _mm256_add_ps (_mm256_set1_ps (0.06681017f), _mm256_mul_ps (x, _mm256_add_ps (_mm256_set1_ps (-0.09402311f), _mm256_mul_ps (x, _mm256_set1_ps (0.02708663f)))));
  return _mm256_add_ps (_mm256_mul_ps (mul_term, _mm256_sqrt_ps (_mm256_add_ps (_mm256_set1_ps (0.89286965f), _mm256_mul_ps (_mm256_set1_ps (-0.89282669f), x)))), add_term);
}

inline __m256
pcl::getAcuteAngle3DAVX (const __m256 &x1, const __m256 &y1, const __m256 &z1, const __m256 &x2, const __m256 &y2, const __m256 &z2)
{
  const __m256 dot_product = _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (x1, x2), _mm256_mul_ps (y1, y2)), _mm256_mul_ps (z1, z2));
  // The andnot-function realizes an abs-operation: the sign bit is removed
  // -0.0f (negative zero) means that all bits are 0, only the sign bit is 1
  return acos_AVX (_mm256_min_ps (_mm256_set1_ps (1.0f), _mm256_andnot_ps (_mm256_set1_ps (-0.0f), dot_product)));
}
#endif // ifdef __AVX__

#ifdef __RVV10__
inline vfloat32m2_t
pcl::acos_RVV_f32m2 (const vfloat32m2_t& x, const std::size_t vl)
{
  // Coefficients (broadcasted)
  const vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2 (1.59121552f, vl);
  const vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2 (-0.15461442f, vl);
  const vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2 (0.05354897f, vl);
  const vfloat32m2_t b0 = __riscv_vfmv_v_f_f32m2 (0.89286965f, vl);
  const vfloat32m2_t b1 = __riscv_vfmv_v_f_f32m2 (-0.89282669f, vl);
  const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2 (0.06681017f, vl);
  const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2 (-0.09402311f, vl);
  const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2 (0.02708663f, vl);

  // mul_term = a0 + x*(a1 + x*a2)
  const vfloat32m2_t mul_term = __riscv_vfmacc_vv_f32m2 (a0, x, __riscv_vfmacc_vv_f32m2 (a1, x, a2, vl), vl);

  // sqrt_term = sqrt(b0 + x*b1)
  const vfloat32m2_t sqrt_term = __riscv_vfsqrt_v_f32m2 (__riscv_vfmacc_vv_f32m2 (b0, x, b1, vl), vl);

  // add_term = c0 + x*(c1 + x*c2)
  const vfloat32m2_t add_term = __riscv_vfmacc_vv_f32m2 (c0, x, __riscv_vfmacc_vv_f32m2 (c1, x, c2, vl), vl);

  // result = mul_term * sqrt_term + add_term
  return __riscv_vfmacc_vv_f32m2 (add_term, mul_term, sqrt_term, vl);
}

inline vfloat32m2_t
pcl::getAcuteAngle3DRVV_f32m2 (const vfloat32m2_t& x1, const vfloat32m2_t& y1, const vfloat32m2_t& z1,
                        const vfloat32m2_t& x2, const vfloat32m2_t& y2, const vfloat32m2_t& z2, const std::size_t vl)
{
  // dot = x1*x2 + y1*y2 + z1*z2
  const vfloat32m2_t dot = __riscv_vfmacc_vv_f32m2 (
      __riscv_vfmacc_vv_f32m2 (
          __riscv_vfmul_vv_f32m2 (x1, x2, vl),
          y1, y2, vl),
      z1, z2, vl);

  // Compute Absolute Value
  // Use vfsgnjx (Floating-point Sign Injection - XOR) with itself.
  const vfloat32m2_t dot_abs = __riscv_vfsgnjx_vv_f32m2 (dot, dot, vl);

  // Clamp to [0, 1]
  const vfloat32m2_t dot_clamped = __riscv_vfmin_vf_f32m2 (dot_abs, 1.0f, vl);

  return acos_RVV_f32m2 (dot_clamped, vl);
}

inline vfloat32m2_t
pcl::atan2_RVV_f32m2 (const vfloat32m2_t& y, const vfloat32m2_t& x, const std::size_t vl)
{
  // Polynomial coefficients (Hastings-style, odd powers), ~0.01 deg max error
  // parms form https://mazzo.li/posts/vectorized-atan2.html
  const float a1 = 0.99997726f;
  const float a3 = -0.33262347f;
  const float a5 = 0.19354346f;
  const float a7 = -0.11643287f;
  const float a9 = 0.05265332f;
  const float a11 = -0.01172120f;
  const float pi = 3.14159265358979323846f;
  const float pi_2 = 1.57079632679489661923f;
  const float tiny = 1e-20f;

  const vfloat32m2_t abs_x = __riscv_vfsgnjx_vv_f32m2 (x, x, vl);
  const vfloat32m2_t abs_y = __riscv_vfsgnjx_vv_f32m2 (y, y, vl);
  // swap when |y| > |x|; vmerge(op1, op2, mask) => mask ? op2 : op1 (match atan2.cpp)
  const vbool16_t swap_mask = __riscv_vmflt_vv_f32m2_b16 (abs_x, abs_y, vl);
  const vfloat32m2_t num = __riscv_vmerge_vvm_f32m2 (y, x, swap_mask, vl);
  const vfloat32m2_t den = __riscv_vmerge_vvm_f32m2 (x, y, swap_mask, vl);
  // Preserve sign of den when clamping (scalar: den = (den>=0)? tiny : -tiny)
  const vfloat32m2_t abs_den = __riscv_vfsgnjx_vv_f32m2 (den, den, vl);
  const vfloat32m2_t den_safe = __riscv_vfsgnj_vv_f32m2 (
      __riscv_vfmax_vf_f32m2 (abs_den, tiny, vl), den, vl);
  vfloat32m2_t atan_input = __riscv_vfdiv_vv_f32m2 (num, den_safe, vl);
  // Clamp to [-1,1] so polynomial stays valid (avoids overflow from float noise)
  atan_input = __riscv_vfmin_vf_f32m2 (__riscv_vfmax_vf_f32m2 (atan_input, -1.0f, vl), 1.0f, vl);

  const vfloat32m2_t x2 = __riscv_vfmul_vv_f32m2 (atan_input, atan_input, vl);
  vfloat32m2_t p = __riscv_vfmv_v_f_f32m2 (a11, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a9, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a7, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a5, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a3, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a1, vl), x2, p, vl);
  vfloat32m2_t result = __riscv_vfmul_vv_f32m2 (atan_input, p, vl);

  // When swapped: adj = +pi/2 when atan_input>=0 (same sign x,y), else -pi/2
  const vbool16_t atan_ge_zero = __riscv_vmfge_vf_f32m2_b16 (atan_input, 0.0f, vl);
  const vfloat32m2_t pi_2_vec = __riscv_vfmv_v_f_f32m2 (pi_2, vl);
  const vfloat32m2_t neg_pi_2 = __riscv_vfmv_v_f_f32m2 (-pi_2, vl);
  const vfloat32m2_t adj = __riscv_vmerge_vvm_f32m2 (neg_pi_2, pi_2_vec, atan_ge_zero, vl);
  result = __riscv_vmerge_vvm_f32m2 (result, __riscv_vfsub_vv_f32m2 (adj, result, vl), swap_mask, vl);

  const vbool16_t x_lt_zero = __riscv_vmflt_vf_f32m2_b16 (x, 0.0f, vl);
  const vbool16_t y_ge_zero = __riscv_vmfge_vf_f32m2_b16 (y, 0.0f, vl);
  const vfloat32m2_t pi_vec = __riscv_vfmv_v_f_f32m2 (pi, vl);
  const vfloat32m2_t neg_pi = __riscv_vfmv_v_f_f32m2 (-pi, vl);
  const vfloat32m2_t add_val = __riscv_vmerge_vvm_f32m2 (neg_pi, pi_vec, y_ge_zero, vl);
  result = __riscv_vmerge_vvm_f32m2 (result, __riscv_vfadd_vv_f32m2 (result, add_val, vl), x_lt_zero, vl);

  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------------
// expf_RVV_f32m2: vectorized float exp using reduction → approximation → reconstruction
// -----------------------------------------------------------------------------
// 约化: x = n*ln2 + r, r ∈ [-ln2/2, ln2/2]. 逼近: exp(r) ≈ P(r) (Remez degree 7).
// 重构: exp(x) = 2^n * P(r). 2^n 用查表 (vluxei32 字节偏移).
// 误差: 相对 std::expf 最大相对误差约 1.5e-6. 系数见 test-rvv/2d/remez_exp.py
//
// 2^n 计算说明:
//   - 约化后 n = round(x/ln2) 是整数，范围约 [-127, 128]（对应 x ∈ [-88, 88]）
//   - 2^n = 2^(n + 127) << 23，n ∈ [-127, 128]
//   - 2^-127 对应 IEEE754 非规格化数：exp=0, mantissa=2^22
// -----------------------------------------------------------------------------
namespace {
  const float kExpfLog2Inv  = 1.4426950408889634f;     // 1 / ln(2)
  const float kExpfLog2Hi   = 0.6931471824645996f;     // ln(2) 高精度部分
  const float kExpfLog2Lo   = -1.904654290582768e-09f; // ln(2) 低精度部分（补偿）
  const float kExpfXMax     = 88.0f;   // 输入上限（exp(88) ≈ 1.6e38，接近 float 上限）
  const float kExpfXMin     = -88.0f;  // 输入下限（exp(-88) ≈ 6e-39，接近 float 下界）
  // 2^-127 对应 IEEE754 非规格化数：exp=0, mantissa=2^22
  const float kExpfTwoToMinus127 = 5.877471754111438e-39f;
  // Remez polynomial for exp(r) on [0, ln(2)], degree 7 (max abs err ~1.85e-10)
  const float kExpfRemezC0  = 9.9999999998e-01f;
  const float kExpfRemezC1  = 1.0000000154e+00f;
  const float kExpfRemezC2  = 4.9999959620e-01f;
  const float kExpfRemezC3  = 1.6667078702e-01f;
  const float kExpfRemezC4  = 4.1645250213e-02f;
  const float kExpfRemezC5  = 8.3952782982e-03f;
  const float kExpfRemezC6  = 1.2887034349e-03f;
  const float kExpfRemezC7  = 2.8147688485e-04f;
}

inline vfloat32m2_t
pcl::expf_RVV_f32m2 (const vfloat32m2_t& x, const std::size_t vl)
{
  // Clamp to avoid overflow/underflow
  vfloat32m2_t vx = __riscv_vfmin_vf_f32m2 (__riscv_vfmax_vf_f32m2 (x, kExpfXMin, vl), kExpfXMax, vl);
  // n = round(x / ln2), flt_n = (float)n
  vfloat32m2_t flt_n = __riscv_vfmul_vf_f32m2 (vx, kExpfLog2Inv, vl);
  vint32m2_t n = __riscv_vfcvt_x_f_v_i32m2 (flt_n, vl);
  flt_n = __riscv_vfcvt_f_x_v_f32m2 (n, vl);
  // r = x - n*log2_hi - n*log2_lo
  vfloat32m2_t r = __riscv_vfnmsub_vf_f32m2 (flt_n, kExpfLog2Hi, vx, vl);
  r = __riscv_vfnmsub_vf_f32m2 (flt_n, kExpfLog2Lo, r, vl);
  // Horner: poly = c7; poly = c6 + r*poly; ... ; exp_r = c0 + r*poly
  vfloat32m2_t poly = __riscv_vfmv_v_f_f32m2 (kExpfRemezC7, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC6, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC5, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC4, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC3, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC2, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC1, vl), r, poly, vl);
  vfloat32m2_t exp_r = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC0, vl), r, poly, vl);
  // 位构造 2^n：normal 情况用 exp_offset<<23，n==-127 时补非规格化 2^-127。
  vint32m2_t exp_offset = __riscv_vadd_vx_i32m2 (n, 127, vl);
  exp_offset = __riscv_vmax_vx_i32m2 (exp_offset, 0, vl);
  exp_offset = __riscv_vmin_vx_i32m2 (exp_offset, 255, vl);
  vuint32m2_t res_bits =
      __riscv_vsll_vx_u32m2 (__riscv_vreinterpret_v_i32m2_u32m2 (exp_offset), 23, vl);
  vfloat32m2_t two_n_normal = __riscv_vreinterpret_v_u32m2_f32m2 (res_bits);
  const vbool16_t is_n_neg127 = __riscv_vmseq_vx_i32m2_b16 (n, -127, vl);
  vfloat32m2_t two_n_sub = __riscv_vfmv_v_f_f32m2 (kExpfTwoToMinus127, vl);
  vfloat32m2_t two_n = __riscv_vmerge_vvm_f32m2 (two_n_normal, two_n_sub, is_n_neg127, vl);
  return __riscv_vfmul_vv_f32m2 (exp_r, two_n, vl);  // exp(x) = 2^n * exp(r)
}
#endif // ifdef __RVV10__

///////////////////////////////////////////////////////////////////////////
/** \brief Scalar kernel: compute sum and sum-of-squares of data[0..n-1]. Used by getMeanStd. */
inline void
getMeanStdKernelStandard (const float* data, std::size_t n, double& sum, double& sq_sum)
{
  sum = 0;
  sq_sum = 0;
  for (std::size_t i = 0; i < n; ++i)
  {
    const float v = data[i];
    sum += v;
    sq_sum += static_cast<double>(v) * v;
  }
}

#if defined(__RVV10__)
/** \brief RVV kernel: compute sum and sum-of-squares of data[0..n-1].
 *
 * \note Precision: accumulation uses \c float (vfloat32m2) and only the final
 *       sums are cast to \c double. The scalar path (\ref getMeanStdKernelStandard)
 *       accumulates in \c double from each \c float sample, so results may differ.
 */
inline void
getMeanStdKernelRVV (const float* data, std::size_t n, double& sum, double& sq_sum)
{
  sum = 0;
  sq_sum = 0;
  std::size_t i = 0;
  const std::size_t max_vl = __riscv_vsetvl_e32m2 (n);
  vfloat32m2_t v_acc_sum = __riscv_vfmv_v_f_f32m2 (0.0f, max_vl);
  vfloat32m2_t v_acc_sq  = __riscv_vfmv_v_f_f32m2 (0.0f, max_vl);
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vfloat32m2_t v = __riscv_vle32_v_f32m2 (data + i, vl);
    v_acc_sum = __riscv_vfadd_vv_f32m2_tu (v_acc_sum, v_acc_sum, v, vl);
    v_acc_sq  = __riscv_vfmacc_vv_f32m2_tu (v_acc_sq, v, v, vl);
    i += vl;
  }
  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  vfloat32m1_t v_sum  = __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_sum, v_zero, max_vl);
  vfloat32m1_t v_sq   = __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_sq,  v_zero, max_vl);
  sum    = static_cast<double>(__riscv_vfmv_f_s_f32m1_f32 (v_sum));
  sq_sum = static_cast<double>(__riscv_vfmv_f_s_f32m1_f32 (v_sq));
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
inline void
pcl::getMeanStd (const std::vector<float> &values, double &mean, double &stddev)
{
  // throw an exception when the input array is empty
  if (values.empty ())
  {
    PCL_THROW_EXCEPTION (BadArgumentException, "Input array must have at least 1 element.");
  }

  // when the array has only one element, mean is the number itself and standard dev is 0
  if (values.size () == 1)
  {
    mean = values.at (0);
    stddev = 0;
    return;
  }

  double sum = 0, sq_sum = 0;
#if defined(__RVV10__)
  getMeanStdKernelRVV (values.data (), values.size (), sum, sq_sum);
#else
  getMeanStdKernelStandard (values.data (), values.size (), sum, sq_sum);
#endif
  mean = sum / static_cast<double>(values.size ());
  double variance = (sq_sum - sum * sum / static_cast<double>(values.size ())) / (static_cast<double>(values.size ()) - 1);
  stddev = sqrt (variance);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// getPointsInBox: __RVV10__ 时 getPointsInBoxRVV（dense 且 n≥16 为向量条带，否则标量），否则 getPointsInBoxStandard。
/** \brief Scalar path for getPointsInBox. Not for direct use; see pcl::getPointsInBox. */
template <typename PointT>
inline int
getPointsInBoxStandard (const pcl::PointCloud<PointT> &cloud,
                        Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt,
                        pcl::Indices &indices)
{
  if (cloud.is_dense)
  {
    int l = 0;
    for (std::size_t i = 0; i < cloud.size (); ++i)
    {
      if (cloud[i].x < min_pt[0] || cloud[i].y < min_pt[1] || cloud[i].z < min_pt[2])
        continue;
      if (cloud[i].x > max_pt[0] || cloud[i].y > max_pt[1] || cloud[i].z > max_pt[2])
        continue;
      indices[l++] = static_cast<pcl::index_t>(i);
    }
    return l;
  }
  int l = 0;
  for (std::size_t i = 0; i < cloud.size (); ++i)
  {
    if (!std::isfinite (cloud[i].x) ||
        !std::isfinite (cloud[i].y) ||
        !std::isfinite (cloud[i].z))
      continue;
    if (cloud[i].x < min_pt[0] || cloud[i].y < min_pt[1] || cloud[i].z < min_pt[2])
      continue;
    if (cloud[i].x > max_pt[0] || cloud[i].y > max_pt[1] || cloud[i].z > max_pt[2])
      continue;
    indices[l++] = static_cast<pcl::index_t>(i);
  }
  return l;
}

#if defined(__RVV10__)
/** \brief RVV path: dense and n≥16 uses strip-mine + vcompress; else getPointsInBoxStandard. */
template <typename PointT>
inline int
getPointsInBoxRVV (const pcl::PointCloud<PointT> &cloud,
                   Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt,
                   pcl::Indices &indices)
{
  const std::size_t n = cloud.size ();
  if (!cloud.is_dense || n < 16)
    return getPointsInBoxStandard (cloud, min_pt, max_pt, indices);

  const std::size_t stride = sizeof (PointT);
  const uint8_t* base = reinterpret_cast<const uint8_t*>(cloud.data ());
  const float min_x = min_pt[0], min_y = min_pt[1], min_z = min_pt[2];
  const float max_x = max_pt[0], max_y = max_pt[1], max_z = max_pt[2];
  int l = 0;
  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const float* ptr_x = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, x));
    const float* ptr_y = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, y));
    const float* ptr_z = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, z));
    const vfloat32m2_t vx = __riscv_vlse32_v_f32m2 (ptr_x, stride, vl);
    const vfloat32m2_t vy = __riscv_vlse32_v_f32m2 (ptr_y, stride, vl);
    const vfloat32m2_t vz = __riscv_vlse32_v_f32m2 (ptr_z, stride, vl);

    vbool16_t in_x = __riscv_vmfge_vf_f32m2_b16 (vx, min_x, vl);
    in_x = __riscv_vmand_mm_b16 (in_x, __riscv_vmfle_vf_f32m2_b16 (vx, max_x, vl), vl);
    vbool16_t in_y = __riscv_vmfge_vf_f32m2_b16 (vy, min_y, vl);
    in_y = __riscv_vmand_mm_b16 (in_y, __riscv_vmfle_vf_f32m2_b16 (vy, max_y, vl), vl);
    vbool16_t in_z = __riscv_vmfge_vf_f32m2_b16 (vz, min_z, vl);
    in_z = __riscv_vmand_mm_b16 (in_z, __riscv_vmfle_vf_f32m2_b16 (vz, max_z, vl), vl);
    vbool16_t mask = __riscv_vmand_mm_b16 (__riscv_vmand_mm_b16 (in_x, in_y, vl), in_z, vl);

    const vuint32m2_t vid = __riscv_vadd_vx_u32m2 (__riscv_vid_v_u32m2 (vl), static_cast<uint32_t> (i), vl);
    const vuint32m2_t compressed = __riscv_vcompress_vm_u32m2 (vid, mask, vl);
    const std::size_t cnt = __riscv_vcpop_m_b16 (mask, vl);
    if (cnt > 0)
    {
      // cnt <= vl <= VLMAX for this strip; vcompress packs the first cnt lanes.
      const std::size_t vl_store = __riscv_vsetvl_e32m2 (cnt);
      std::uint32_t* const out_u32 =
          reinterpret_cast<std::uint32_t*> (indices.data () + l);
      __riscv_vse32_v_u32m2 (out_u32, compressed, vl_store);
      l += static_cast<int>(cnt);
    }
    i += vl;
  }
  return l;
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline void
pcl::getPointsInBox (const pcl::PointCloud<PointT> &cloud,
                     Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt,
                     Indices &indices)
{
  indices.resize (cloud.size ());
  int l;
#if defined(__RVV10__)
  l = getPointsInBoxRVV (cloud, min_pt, max_pt, indices);
#else
  l = getPointsInBoxStandard (cloud, min_pt, max_pt, indices);
#endif
  indices.resize (l);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// getMaxDistance: Standard = 原本实现 (getVector3fMap + .norm()). 分发: __RVV10__ 时 RVV，否则 Standard。
template<typename PointT>
inline void setMaxPt (const pcl::PointCloud<PointT> &cloud, int max_idx, Eigen::Vector4f &max_pt)
{
  if (max_idx != -1)
    max_pt = cloud[max_idx].getVector4fMap ();
  else
    max_pt = Eigen::Vector4f (std::numeric_limits<float>::quiet_NaN (), std::numeric_limits<float>::quiet_NaN (),
                              std::numeric_limits<float>::quiet_NaN (), std::numeric_limits<float>::quiet_NaN ());
}

template<typename PointT>
inline void setMaxPtFromIndices (const pcl::PointCloud<PointT> &cloud, const pcl::Indices &indices, int max_idx, Eigen::Vector4f &max_pt)
{
  if (max_idx != -1)
    max_pt = cloud[indices[static_cast<std::size_t>(max_idx)]].getVector4fMap ();
  else
    max_pt = Eigen::Vector4f (std::numeric_limits<float>::quiet_NaN (), std::numeric_limits<float>::quiet_NaN (),
                              std::numeric_limits<float>::quiet_NaN (), std::numeric_limits<float>::quiet_NaN ());
}

// Standard: L2 via getVector3fMap() and .norm()
template<typename PointT>
inline void getMaxDistanceStandard (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
  float max_dist = std::numeric_limits<float>::lowest();
  int max_idx = -1;
  float dist;
  const Eigen::Vector3f pivot_pt3 = pivot_pt.head<3>();

  if (cloud.is_dense)
  {
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
      pcl::Vector3fMapConst pt = cloud[i].getVector3fMap();
      dist = (pivot_pt3 - pt).norm();
      if (dist > max_dist)
      {
        max_idx = static_cast<int>(i);
        max_dist = dist;
      }
    }
  }
  else
  {
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
      if (!std::isfinite(cloud[i].x) || !std::isfinite(cloud[i].y) ||
          !std::isfinite(cloud[i].z))
        continue;
      pcl::Vector3fMapConst pt = cloud[i].getVector3fMap();
      dist = (pivot_pt3 - pt).norm();
      if (dist > max_dist)
      {
        max_idx = static_cast<int>(i);
        max_dist = dist;
      }
    }
  }
  setMaxPt(cloud, max_idx, max_pt);
}

// RVV: when __RVV10__ and dense and n>=16 use RVV, else getMaxDistanceStandard. L2²; argmax(L2)=argmax(L2²).
#if defined(__RVV10__)
template<typename PointT>
inline void getMaxDistanceRVV (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
  const float px = pivot_pt[0], py = pivot_pt[1], pz = pivot_pt[2];
  const std::size_t n = cloud.size ();

  if (!cloud.is_dense || n < 16)
  {
    getMaxDistanceStandard (cloud, pivot_pt, max_pt);
    return;
  }

  const std::size_t stride = sizeof(PointT);
  const uint8_t* base = reinterpret_cast<const uint8_t*>(cloud.data());
  float max_chunk = -1.0f;
  int idx_chunk = -1;
  std::size_t i = 0;
  while (i < n) {
    const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
    const float* ptr_x =
        reinterpret_cast<const float*>(base + i * stride + offsetof(PointT, x));
    const float* ptr_y =
        reinterpret_cast<const float*>(base + i * stride + offsetof(PointT, y));
    const float* ptr_z =
        reinterpret_cast<const float*>(base + i * stride + offsetof(PointT, z));
    const vfloat32m2_t vx = __riscv_vlse32_v_f32m2(ptr_x, stride, vl);
    const vfloat32m2_t vy = __riscv_vlse32_v_f32m2(ptr_y, stride, vl);
    const vfloat32m2_t vz = __riscv_vlse32_v_f32m2(ptr_z, stride, vl);
    const vfloat32m2_t v_dx = __riscv_vfrsub_vf_f32m2(vx, px, vl);
    const vfloat32m2_t v_dy = __riscv_vfrsub_vf_f32m2(vy, py, vl);
    const vfloat32m2_t v_dz = __riscv_vfrsub_vf_f32m2(vz, pz, vl);
    const vfloat32m2_t v_d2 = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmacc_vv_f32m2(__riscv_vfmul_vv_f32m2(v_dx, v_dx, vl), v_dy, v_dy, vl),
        v_dz,
        v_dz,
        vl);
    const vfloat32m1_t v_max =
        __riscv_vfredmax_vs_f32m2_f32m1(v_d2, __riscv_vfmv_s_f_f32m1(-1.0f, 1), vl);
    const float chunk_max = __riscv_vfmv_f_s_f32m1_f32(v_max);
    if (chunk_max > max_chunk) {
      const vfloat32m2_t v_broadcast = __riscv_vfmv_v_f_f32m2(chunk_max, vl);
      const vbool16_t mask = __riscv_vmfeq_vv_f32m2_b16(v_d2, v_broadcast, vl);
      const vuint32m2_t vid =
          __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl), static_cast<uint32_t>(i), vl);
      const vuint32m2_t comp = __riscv_vcompress_vm_u32m2(vid, mask, vl);
      idx_chunk = static_cast<int>(__riscv_vmv_x_s_u32m2_u32(comp));
      max_chunk = chunk_max;
    }
    i += vl;
  }
  int max_idx = -1;
  if (idx_chunk >= 0)
    max_idx = idx_chunk;
  setMaxPt (cloud, max_idx, max_pt);
}
#endif

// Standard (indices version)
template<typename PointT>
inline void getMaxDistanceStandard (const pcl::PointCloud<PointT> &cloud, const pcl::Indices &indices,
                                        const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
  float max_dist = std::numeric_limits<float>::lowest();
  int max_idx = -1;
  float dist;
  const Eigen::Vector3f pivot_pt3 = pivot_pt.head<3>();

  if (cloud.is_dense)
  {
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
      pcl::Vector3fMapConst pt = cloud[indices[i]].getVector3fMap();
      dist = (pivot_pt3 - pt).norm();
      if (dist > max_dist)
      {
        max_idx = static_cast<int>(i);
        max_dist = dist;
      }
    }
  }
  else {
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
      if (!std::isfinite(cloud[indices[i]].x) || !std::isfinite(cloud[indices[i]].y) ||
          !std::isfinite(cloud[indices[i]].z))
        continue;
      pcl::Vector3fMapConst pt = cloud[indices[i]].getVector3fMap();
      dist = (pivot_pt3 - pt).norm();
      if (dist > max_dist)
      {
        max_idx = static_cast<int>(i);
        max_dist = dist;
      }
    }
  }
  setMaxPtFromIndices(cloud, indices, max_idx, max_pt);
}

// RVV with gather for indices (when __RVV10__ and dense and n>=16)
#if defined(__RVV10__)
template<typename PointT>
inline void getMaxDistanceRVV (const pcl::PointCloud<PointT> &cloud, const pcl::Indices &indices,
                                         const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
  const float px = pivot_pt[0], py = pivot_pt[1], pz = pivot_pt[2];
  const std::size_t n = indices.size ();

  // 与「is_dense → (n>=16 ? RVV : std) : std」等价；早退减少嵌套。
  if (!cloud.is_dense || n < 16)
  {
    getMaxDistanceStandard (cloud, indices, pivot_pt, max_pt);
    return;
  }

  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(cloud.data ());
  const uint32_t* indices_ptr = reinterpret_cast<const uint32_t*>(indices.data ());
  float max_chunk = -1.0f;
  int idx_chunk = -1;
  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2 (indices_ptr + i, vl);
    const vuint32m2_t v_off = __riscv_vmul_vx_u32m2 (v_idx, sizeof (PointT), vl);
    const vfloat32m2_t vx = __riscv_vluxei32_v_f32m2 (
        reinterpret_cast<const float*>(points_base + offsetof (PointT, x)), v_off, vl);
    const vfloat32m2_t vy = __riscv_vluxei32_v_f32m2 (
        reinterpret_cast<const float*>(points_base + offsetof (PointT, y)), v_off, vl);
    const vfloat32m2_t vz = __riscv_vluxei32_v_f32m2 (
        reinterpret_cast<const float*>(points_base + offsetof (PointT, z)), v_off, vl);
    const vfloat32m2_t v_dx = __riscv_vfrsub_vf_f32m2 (vx, px, vl);
    const vfloat32m2_t v_dy = __riscv_vfrsub_vf_f32m2 (vy, py, vl);
    const vfloat32m2_t v_dz = __riscv_vfrsub_vf_f32m2 (vz, pz, vl);
    const vfloat32m2_t v_d2 = __riscv_vfmacc_vv_f32m2 (
        __riscv_vfmacc_vv_f32m2 (__riscv_vfmul_vv_f32m2 (v_dx, v_dx, vl), v_dy, v_dy, vl),
        v_dz,
        v_dz,
        vl);
    const vfloat32m1_t v_max =
        __riscv_vfredmax_vs_f32m2_f32m1 (v_d2, __riscv_vfmv_s_f_f32m1 (-1.0f, 1), vl);
    const float chunk_max = __riscv_vfmv_f_s_f32m1_f32 (v_max);
    if (chunk_max > max_chunk)
    {
      const vfloat32m2_t v_broadcast = __riscv_vfmv_v_f_f32m2 (chunk_max, vl);
      const vbool16_t mask = __riscv_vmfeq_vv_f32m2_b16 (v_d2, v_broadcast, vl);
      const vuint32m2_t vid =
          __riscv_vadd_vx_u32m2 (__riscv_vid_v_u32m2 (vl), static_cast<uint32_t> (i), vl);
      const vuint32m2_t comp = __riscv_vcompress_vm_u32m2 (vid, mask, vl);
      idx_chunk = static_cast<int>(__riscv_vmv_x_s_u32m2_u32 (comp));
      max_chunk = chunk_max;
    }
    i += vl;
  }
  int max_idx = -1;
  if (idx_chunk >= 0)
    max_idx = idx_chunk;
  setMaxPtFromIndices (cloud, indices, max_idx, max_pt);
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> inline void
pcl::getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
#if defined(__RVV10__)
  getMaxDistanceRVV (cloud, pivot_pt, max_pt);
#else
  getMaxDistanceStandard (cloud, pivot_pt, max_pt);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> inline void
pcl::getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
                     const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
#if defined(__RVV10__)
  getMaxDistanceRVV (cloud, indices, pivot_pt, max_pt);
#else
  getMaxDistanceStandard (cloud, indices, pivot_pt, max_pt);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////
// getMinMax3D: Standard = 原本实现 (getVector4fMap + cwiseMin/cwiseMax). 分发: __RVV10__ 时 RVV，否则 Standard。
// Standard: 上游 PCL 原实现; RVV: __RVV10__ 且 dense 且 n>=16 时 vfredmin/vfredmax。
// Standard (original implementation from upstream PCL)
template <typename PointT>
inline void getMinMax3DStandard (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
  min_pt.setConstant (std::numeric_limits<float>::max ());
  max_pt.setConstant (std::numeric_limits<float>::lowest ());
  if (cloud.is_dense)
  {
    for (const auto& point : cloud.points)
    {
      const pcl::Vector4fMapConst pt = point.getVector4fMap ();
      min_pt = min_pt.cwiseMin (pt);
      max_pt = max_pt.cwiseMax (pt);
    }
  }
  else
  {
    for (const auto& point : cloud.points)
    {
      if (!std::isfinite (point.x) || !std::isfinite (point.y) || !std::isfinite (point.z))
        continue;
      const pcl::Vector4fMapConst pt = point.getVector4fMap ();
      min_pt = min_pt.cwiseMin (pt);
      max_pt = max_pt.cwiseMax (pt);
    }
  }
}

template <typename PointT>
inline void getMinMax3DStandard (const pcl::PointCloud<PointT> &cloud, const pcl::Indices &indices,
                                Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
  min_pt.setConstant (std::numeric_limits<float>::max ());
  max_pt.setConstant (std::numeric_limits<float>::lowest ());
  if (cloud.is_dense)
  {
    for (const auto &index : indices)
    {
      const pcl::Vector4fMapConst pt = cloud[index].getVector4fMap ();
      min_pt = min_pt.cwiseMin (pt);
      max_pt = max_pt.cwiseMax (pt);
    }
  }
  else
  {
    for (const auto &index : indices)
    {
      if (!std::isfinite (cloud[index].x) || !std::isfinite (cloud[index].y) || !std::isfinite (cloud[index].z))
        continue;
      const pcl::Vector4fMapConst pt = cloud[index].getVector4fMap ();
      min_pt = min_pt.cwiseMin (pt);
      max_pt = max_pt.cwiseMax (pt);
    }
  }
}

#if defined(__RVV10__)
template <typename PointT>
inline void getMinMax3DRVV (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
  const std::size_t n = cloud.size ();

  if (!cloud.is_dense || n < 16)
  {
    getMinMax3DStandard (cloud, min_pt, max_pt);
    return;
  }

  const std::size_t vlmax = __riscv_vsetvl_e32m2 (static_cast<std::size_t> (-1));
  const float init_min = std::numeric_limits<float>::max ();
  const float init_max = std::numeric_limits<float>::lowest ();
  const std::size_t stride = sizeof (PointT);
  const uint8_t* base = reinterpret_cast<const uint8_t*>(cloud.data ());

  // 循环内按 lane 做 vfmin/vfmax 归并；_tu 避免末段 vl<vlmax 时 TA 破坏高 lane 上已有极值。循环外各维一次 vfred。
  vfloat32m2_t v_acc_min_x = __riscv_vfmv_v_f_f32m2 (init_min, vlmax);
  vfloat32m2_t v_acc_min_y = __riscv_vfmv_v_f_f32m2 (init_min, vlmax);
  vfloat32m2_t v_acc_min_z = __riscv_vfmv_v_f_f32m2 (init_min, vlmax);
  vfloat32m2_t v_acc_max_x = __riscv_vfmv_v_f_f32m2 (init_max, vlmax);
  vfloat32m2_t v_acc_max_y = __riscv_vfmv_v_f_f32m2 (init_max, vlmax);
  vfloat32m2_t v_acc_max_z = __riscv_vfmv_v_f_f32m2 (init_max, vlmax);

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const float* ptr_x =
        reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, x));
    const float* ptr_y =
        reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, y));
    const float* ptr_z =
        reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, z));
    const vfloat32m2_t vx = __riscv_vlse32_v_f32m2 (ptr_x, stride, vl);
    const vfloat32m2_t vy = __riscv_vlse32_v_f32m2 (ptr_y, stride, vl);
    const vfloat32m2_t vz = __riscv_vlse32_v_f32m2 (ptr_z, stride, vl);
    v_acc_min_x = __riscv_vfmin_vv_f32m2_tu (v_acc_min_x, v_acc_min_x, vx, vl);
    v_acc_min_y = __riscv_vfmin_vv_f32m2_tu (v_acc_min_y, v_acc_min_y, vy, vl);
    v_acc_min_z = __riscv_vfmin_vv_f32m2_tu (v_acc_min_z, v_acc_min_z, vz, vl);
    v_acc_max_x = __riscv_vfmax_vv_f32m2_tu (v_acc_max_x, v_acc_max_x, vx, vl);
    v_acc_max_y = __riscv_vfmax_vv_f32m2_tu (v_acc_max_y, v_acc_max_y, vy, vl);
    v_acc_max_z = __riscv_vfmax_vv_f32m2_tu (v_acc_max_z, v_acc_max_z, vz, vl);
    i += vl;
  }

  const vfloat32m1_t red_min_seed = __riscv_vfmv_s_f_f32m1 (init_min, 1);
  const vfloat32m1_t red_max_seed = __riscv_vfmv_s_f_f32m1 (init_max, 1);
  min_pt[0] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmin_vs_f32m2_f32m1 (v_acc_min_x, red_min_seed, vlmax));
  min_pt[1] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmin_vs_f32m2_f32m1 (v_acc_min_y, red_min_seed, vlmax));
  min_pt[2] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmin_vs_f32m2_f32m1 (v_acc_min_z, red_min_seed, vlmax));
  min_pt[3] = 0.0f;
  max_pt[0] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmax_vs_f32m2_f32m1 (v_acc_max_x, red_max_seed, vlmax));
  max_pt[1] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmax_vs_f32m2_f32m1 (v_acc_max_y, red_max_seed, vlmax));
  max_pt[2] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmax_vs_f32m2_f32m1 (v_acc_max_z, red_max_seed, vlmax));
  max_pt[3] = 0.0f;
}

template <typename PointT>
inline void getMinMax3DRVV (const pcl::PointCloud<PointT> &cloud, const pcl::Indices &indices,
                            Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
  const std::size_t n = indices.size ();

  if (!cloud.is_dense || n < 16)
  {
    getMinMax3DStandard (cloud, indices, min_pt, max_pt);
    return;
  }

  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(cloud.data ());
  const uint32_t* indices_ptr = reinterpret_cast<const uint32_t*>(indices.data ());
  const std::size_t vlmax = __riscv_vsetvl_e32m2 (static_cast<std::size_t> (-1));
  const float init_min = std::numeric_limits<float>::max ();
  const float init_max = std::numeric_limits<float>::lowest ();

  // 同 dense 版；vfmin/vfmax 用 _tu 保证尾段不污染未参与本拍比较的 lane。
  vfloat32m2_t v_acc_min_x = __riscv_vfmv_v_f_f32m2 (init_min, vlmax);
  vfloat32m2_t v_acc_min_y = __riscv_vfmv_v_f_f32m2 (init_min, vlmax);
  vfloat32m2_t v_acc_min_z = __riscv_vfmv_v_f_f32m2 (init_min, vlmax);
  vfloat32m2_t v_acc_max_x = __riscv_vfmv_v_f_f32m2 (init_max, vlmax);
  vfloat32m2_t v_acc_max_y = __riscv_vfmv_v_f_f32m2 (init_max, vlmax);
  vfloat32m2_t v_acc_max_z = __riscv_vfmv_v_f_f32m2 (init_max, vlmax);
  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2 (indices_ptr + i, vl);
    const vuint32m2_t v_off = __riscv_vmul_vx_u32m2 (v_idx, sizeof (PointT), vl);
    const vfloat32m2_t vx = __riscv_vluxei32_v_f32m2 (
        reinterpret_cast<const float*>(points_base + offsetof (PointT, x)), v_off, vl);
    const vfloat32m2_t vy = __riscv_vluxei32_v_f32m2 (
        reinterpret_cast<const float*>(points_base + offsetof (PointT, y)), v_off, vl);
    const vfloat32m2_t vz = __riscv_vluxei32_v_f32m2 (
        reinterpret_cast<const float*>(points_base + offsetof (PointT, z)), v_off, vl);
    v_acc_min_x = __riscv_vfmin_vv_f32m2_tu (v_acc_min_x, v_acc_min_x, vx, vl);
    v_acc_min_y = __riscv_vfmin_vv_f32m2_tu (v_acc_min_y, v_acc_min_y, vy, vl);
    v_acc_min_z = __riscv_vfmin_vv_f32m2_tu (v_acc_min_z, v_acc_min_z, vz, vl);
    v_acc_max_x = __riscv_vfmax_vv_f32m2_tu (v_acc_max_x, v_acc_max_x, vx, vl);
    v_acc_max_y = __riscv_vfmax_vv_f32m2_tu (v_acc_max_y, v_acc_max_y, vy, vl);
    v_acc_max_z = __riscv_vfmax_vv_f32m2_tu (v_acc_max_z, v_acc_max_z, vz, vl);
    i += vl;
  }
  const vfloat32m1_t red_min_seed = __riscv_vfmv_s_f_f32m1 (init_min, 1);
  const vfloat32m1_t red_max_seed = __riscv_vfmv_s_f_f32m1 (init_max, 1);
  min_pt[0] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmin_vs_f32m2_f32m1 (v_acc_min_x, red_min_seed, vlmax));
  min_pt[1] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmin_vs_f32m2_f32m1 (v_acc_min_y, red_min_seed, vlmax));
  min_pt[2] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmin_vs_f32m2_f32m1 (v_acc_min_z, red_min_seed, vlmax));
  min_pt[3] = 0.0f;
  max_pt[0] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmax_vs_f32m2_f32m1 (v_acc_max_x, red_max_seed, vlmax));
  max_pt[1] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmax_vs_f32m2_f32m1 (v_acc_max_y, red_max_seed, vlmax));
  max_pt[2] = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredmax_vs_f32m2_f32m1 (v_acc_max_z, red_max_seed, vlmax));
  max_pt[3] = 0.0f;
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline void
pcl::getMinMax3D (const pcl::PointCloud<PointT> &cloud, PointT &min_pt, PointT &max_pt)
{
  Eigen::Vector4f min_p, max_p;
  pcl::getMinMax3D (cloud, min_p, max_p);
  min_pt.x = min_p[0]; min_pt.y = min_p[1]; min_pt.z = min_p[2];
  max_pt.x = max_p[0]; max_pt.y = max_p[1]; max_pt.z = max_p[2];
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline void
pcl::getMinMax3D (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
#if defined(__RVV10__)
  getMinMax3DRVV (cloud, min_pt, max_pt);
#else
  getMinMax3DStandard (cloud, min_pt, max_pt);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline void
pcl::getMinMax3D (const pcl::PointCloud<PointT> &cloud, const pcl::PointIndices &indices,
                  Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
  pcl::getMinMax3D (cloud, indices.indices, min_pt, max_pt);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline void
pcl::getMinMax3D (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
                  Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
#if defined(__RVV10__)
  getMinMax3DRVV (cloud, indices, min_pt, max_pt);
#else
  getMinMax3DStandard (cloud, indices, min_pt, max_pt);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline double
pcl::getCircumcircleRadius (const PointT &pa, const PointT &pb, const PointT &pc)
{
  Eigen::Vector4f p1 (pa.x, pa.y, pa.z, 0);
  Eigen::Vector4f p2 (pb.x, pb.y, pb.z, 0);
  Eigen::Vector4f p3 (pc.x, pc.y, pc.z, 0);

  double p2p1 = (p2 - p1).norm (), p3p2 = (p3 - p2).norm (), p1p3 = (p1 - p3).norm ();
  // Calculate the area of the triangle using Heron's formula
  // (https://en.wikipedia.org/wiki/Heron's_formula)
  double semiperimeter = (p2p1 + p3p2 + p1p3) / 2.0;
  double area = sqrt (semiperimeter * (semiperimeter - p2p1) * (semiperimeter - p3p2) * (semiperimeter - p1p3));
  // Compute the radius of the circumscribed circle
  return ((p2p1 * p3p2 * p1p3) / (4.0 * area));
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Based on a search in the current PCL tree, this function does not appear to
// be used inside the core library; Could it be a legacy helper tool for
// histogram-like point types that support [] operator access?
template <typename PointT> inline void
pcl::getMinMax (const PointT &histogram, int len, float &min_p, float &max_p)
{
  min_p = std::numeric_limits<float>::max();
  max_p = std::numeric_limits<float>::lowest();

  for (int i = 0; i < len; ++i)
  {
    min_p = (histogram[i] > min_p) ? min_p : histogram[i];
    max_p = (histogram[i] < max_p) ? max_p : histogram[i];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
// calculatePolygonArea: Standard = 原本实现 (getVector3fMap + cross); RVV 当 __RVV10__ 且 n>=16 时 stride-load + 向量叉积归约；否则 Standard.
template <typename PointT>
inline float calculatePolygonAreaStandard (const pcl::PointCloud<PointT> &polygon)
{
  float area = 0.0f;
  const int num_points = static_cast<int>(polygon.size ());
  Eigen::Vector3f va, vb, res;
  res(0) = res(1) = res(2) = 0.0f;
  for (int i = 0; i < num_points; ++i)
  {
    int j = (i + 1) % num_points;
    va = polygon[i].getVector3fMap ();
    vb = polygon[j].getVector3fMap ();
    res += va.cross (vb);
  }
  area = res.norm ();
  return (area * 0.5f);
}

#if defined(__RVV10__)
template <typename PointT>
inline float calculatePolygonAreaRVV (const pcl::PointCloud<PointT> &polygon)
{
  const int num_points = static_cast<int>(polygon.size ());
  if (num_points < 16)
    return calculatePolygonAreaStandard (polygon);

  const std::size_t n = static_cast<std::size_t>(num_points);
  const std::size_t stride = sizeof (PointT);
  const uint8_t* base = reinterpret_cast<const uint8_t*>(polygon.data ());
  const std::size_t n_pairs = n - 1;
  const std::size_t vlmax = __riscv_vsetvl_e32m2 (static_cast<std::size_t> (-1));
  vfloat32m2_t v_acc_x = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc_y = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc_z = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);

  // Pairs (0,1)..(n-2,n-1): per-lane 累加后 vfredusum；vfadd 用 _tu 避免末段 vl<vlmax 时尾部 lane 被 TA 污染。
  std::size_t i = 0;
  while (i < n_pairs)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n_pairs - i);
    const float* ptr_ax = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, x));
    const float* ptr_ay = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, y));
    const float* ptr_az = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, z));
    const float* ptr_bx = reinterpret_cast<const float*>(base + (i + 1) * stride + offsetof (PointT, x));
    const float* ptr_by = reinterpret_cast<const float*>(base + (i + 1) * stride + offsetof (PointT, y));
    const float* ptr_bz = reinterpret_cast<const float*>(base + (i + 1) * stride + offsetof (PointT, z));
    const vfloat32m2_t ax = __riscv_vlse32_v_f32m2 (ptr_ax, stride, vl);
    const vfloat32m2_t ay = __riscv_vlse32_v_f32m2 (ptr_ay, stride, vl);
    const vfloat32m2_t az = __riscv_vlse32_v_f32m2 (ptr_az, stride, vl);
    const vfloat32m2_t bx = __riscv_vlse32_v_f32m2 (ptr_bx, stride, vl);
    const vfloat32m2_t by = __riscv_vlse32_v_f32m2 (ptr_by, stride, vl);
    const vfloat32m2_t bz = __riscv_vlse32_v_f32m2 (ptr_bz, stride, vl);
    // vfmsac.vv：vd = vs1*vs2 - vd（规范正文，非 vd - vs1*vs2；后者为 vfnmsac）。用 vd=az*by 再 vfmsac(_, ay, bz) 得 ay*bz - az*by = (a×b)_x。
    const vfloat32m2_t cx = __riscv_vfmsac_vv_f32m2 (__riscv_vfmul_vv_f32m2 (az, by, vl), ay, bz, vl);
    const vfloat32m2_t cy = __riscv_vfmsac_vv_f32m2 (__riscv_vfmul_vv_f32m2 (ax, bz, vl), az, bx, vl);
    const vfloat32m2_t cz = __riscv_vfmsac_vv_f32m2 (__riscv_vfmul_vv_f32m2 (ay, bx, vl), ax, by, vl);
    v_acc_x = __riscv_vfadd_vv_f32m2_tu (v_acc_x, v_acc_x, cx, vl);
    v_acc_y = __riscv_vfadd_vv_f32m2_tu (v_acc_y, v_acc_y, cy, vl);
    v_acc_z = __riscv_vfadd_vv_f32m2_tu (v_acc_z, v_acc_z, cz, vl);
    i += vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  float rx = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_x, v_zero, vlmax));
  float ry = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_y, v_zero, vlmax));
  float rz = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_z, v_zero, vlmax));

  // Last pair (n-1, 0)
  const float ax = polygon[n - 1].x, ay = polygon[n - 1].y, az = polygon[n - 1].z;
  const float bx = polygon[0].x, by = polygon[0].y, bz = polygon[0].z;
  rx += ay * bz - az * by;
  ry += az * bx - ax * bz;
  rz += ax * by - ay * bx;
  const float area = std::sqrt (rx * rx + ry * ry + rz * rz) * 0.5f;
  return area;
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> inline float
pcl::calculatePolygonArea (const pcl::PointCloud<PointT> &polygon)
{
#if defined(__RVV10__)
  return calculatePolygonAreaRVV (polygon);
#else
  return calculatePolygonAreaStandard (polygon);
#endif
}

#endif  //#ifndef PCL_COMMON_IMPL_H_

