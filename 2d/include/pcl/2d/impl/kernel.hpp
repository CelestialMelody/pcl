/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
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
 */

#pragma once

#include <pcl/2d/kernel.h>
#if defined(__RVV10__)
#include <riscv_vector.h>
#include <pcl/common/common.h>
#endif

namespace pcl {

#if defined(__RVV10__)
static float
sumReduceRVV(const std::vector<float>& buf, std::size_t n)
{
  const std::size_t max_vl = __riscv_vsetvl_e32m2(n);
  vfloat32m2_t v_acc = __riscv_vfmv_v_f_f32m2(0.0f, max_vl);
  std::size_t j0 = 0;
  while (j0 < n) {
    std::size_t vl = __riscv_vsetvl_e32m2(n - j0);
    vfloat32m2_t v_buf = __riscv_vle32_v_f32m2(buf.data() + j0, vl);
    v_acc = __riscv_vfadd_vv_f32m2(v_acc, v_buf, vl);
    j0 += vl;
  }
  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t v_sum =
      __riscv_vfredosum_vs_f32m2_f32m1(v_acc, v_zero, max_vl);
  return __riscv_vfmv_f_s_f32m1_f32(v_sum);
}

template <typename PointT>
static void
normalizeAndWriteBackRVV(std::vector<float>& buf, float sum, pcl::PointCloud<PointT>& kernel, std::size_t n)
{
  // RVV: normalize buf (contiguous), then strided store to kernel[].intensity
  std::size_t j0 = 0;
  while (j0 < n) {
    std::size_t vl = __riscv_vsetvl_e32m2(n - j0);
    vfloat32m2_t v_buf = __riscv_vle32_v_f32m2(buf.data() + j0, vl);
    v_buf = __riscv_vfdiv_vf_f32m2(v_buf, sum, vl);
    __riscv_vse32_v_f32m2(buf.data() + j0, v_buf, vl);
    j0 += vl;
  }
  const std::size_t point_stride = sizeof(PointT);
  float* intensity_base = &kernel[0].intensity;
  j0 = 0;
  while (j0 < n) {
    std::size_t vl = __riscv_vsetvl_e32m2(n - j0);
    vfloat32m2_t v_vals = __riscv_vle32_v_f32m2(buf.data() + j0, vl);
    float* chunk_base = reinterpret_cast<float*>(reinterpret_cast<char*>(intensity_base) + j0 * point_stride);
    __riscv_vsse32_v_f32m2(chunk_base, point_stride, v_vals, vl);
    j0 += vl;
  }
}
#endif

// Compute Gaussian kernel values (scalar path): buf[idx] = exp(-(iks²+jks²)/sigma_sqr)
static void
computeGaussianKernelValues(std::vector<float>& buf, int kernel_size, float sigma_sqr, int n)
{
  for (int idx = 0; idx < n; ++idx) {
    const int i = idx / kernel_size;
    const int j = idx % kernel_size;
    const int iks = i - kernel_size / 2;
    const int jks = j - kernel_size / 2;
    buf[static_cast<std::size_t>(idx)] = std::exp(
        static_cast<float>(-static_cast<double>(iks * iks + jks * jks) / sigma_sqr));
  }
}

// Compute LoG kernel values (scalar path): buf[idx] = (1 - temp) * exp(-temp), temp = (iks²+jks²)/sigma_sqr
static void
computeLoGKernelValues(std::vector<float>& buf, int kernel_size, float sigma_sqr, int n)
{
  for (int idx = 0; idx < n; ++idx) {
    const int i = idx / kernel_size;
    const int j = idx % kernel_size;
    const int iks = i - kernel_size / 2;
    const int jks = j - kernel_size / 2;
    const float temp = static_cast<float>(static_cast<double>(iks * iks + jks * jks) / sigma_sqr);
    buf[static_cast<std::size_t>(idx)] = (1.0f - temp) * std::exp(-temp);
  }
}

#if defined(__RVV10__)
// RVV path: fill buf with exp(args) using pcl::expf_RVV_f32m2.
static void
computeGaussianKernelValuesRVV(std::vector<float>& buf, int kernel_size, float sigma_sqr, int n)
{
  const std::size_t un = static_cast<std::size_t>(n);
  // 直接在栈上分配。假设目标硬件 VLEN 最大不超过 1024-bit，e32m2 对应最大 64 个 float。
  // 留足冗余，分配 128 个 float（512B），避免 heap / TLS 带来的固定开销与隐式状态。
  constexpr std::size_t MAX_SAFE_VL = 128;
  float tmp[MAX_SAFE_VL];
  std::size_t j0 = 0;

  const float inv_sigma_sqr = -1.0f / sigma_sqr;
  const int half_size = kernel_size / 2;

  while (j0 < un) {
    std::size_t vl = __riscv_vsetvl_e32m2(un - j0);
    if (vl > MAX_SAFE_VL) {
      // 未来如果硬件向量过长，这里限制 vl，避免栈缓冲越界。
      vl = __riscv_vsetvl_e32m2(MAX_SAFE_VL);
    }

    // Build (ii, jj) incrementally to avoid costly per-element division/modulo.
    int ii = static_cast<int>(j0) / kernel_size;
    int jj = static_cast<int>(j0) - ii * kernel_size;

    for (std::size_t i = 0; i < vl; ++i) {
      const int iks = ii - half_size;
      const int jks = jj - half_size;
      tmp[i] = static_cast<float>(iks * iks + jks * jks) * inv_sigma_sqr;

      ++jj;
      if (jj == kernel_size) {
        jj = 0;
        ++ii;
      }
    }

    vfloat32m2_t v_arg = __riscv_vle32_v_f32m2(tmp, vl);
    vfloat32m2_t v_exp = pcl::expf_RVV_f32m2(v_arg, vl);
    __riscv_vse32_v_f32m2(buf.data() + j0, v_exp, vl);
    j0 += vl;
  }
}

// RVV path: LoG buf[idx] = (1 - temp) * exp(-temp)
static void
computeLoGKernelValuesRVV(std::vector<float>& buf, int kernel_size, float sigma_sqr, int n)
{
  const std::size_t un = static_cast<std::size_t>(n);
  // 直接在栈上分配（见 computeGaussianKernelValuesRVV 的说明）。
  constexpr std::size_t MAX_SAFE_VL = 128;
  float tmp_arg[MAX_SAFE_VL];
  float tmp_one_minus[MAX_SAFE_VL];
  std::size_t j0 = 0;

  const float inv_sigma_sqr = 1.0f / sigma_sqr;
  const int half_size = kernel_size / 2;

  while (j0 < un) {
    std::size_t vl = __riscv_vsetvl_e32m2(un - j0);
    if (vl > MAX_SAFE_VL) {
      vl = __riscv_vsetvl_e32m2(MAX_SAFE_VL);
    }
    // Build (ii, jj) incrementally to avoid costly per-element division/modulo.
    int ii = static_cast<int>(j0) / kernel_size;
    int jj = static_cast<int>(j0) - ii * kernel_size;

    for (std::size_t i = 0; i < vl; ++i) {
      const int iks = ii - half_size;
      const int jks = jj - half_size;
      const float temp = static_cast<float>(iks * iks + jks * jks) * inv_sigma_sqr;
      tmp_arg[i] = -temp;
      tmp_one_minus[i] = 1.0f - temp;

      ++jj;
      if (jj == kernel_size) {
        jj = 0;
        ++ii;
      }
    }

    vfloat32m2_t v_arg = __riscv_vle32_v_f32m2(tmp_arg, vl);
    vfloat32m2_t v_exp = pcl::expf_RVV_f32m2(v_arg, vl);
    vfloat32m2_t v_one_minus = __riscv_vle32_v_f32m2(tmp_one_minus, vl);

    __riscv_vse32_v_f32m2(buf.data() + j0, __riscv_vfmul_vv_f32m2(v_exp, v_one_minus, vl), vl);
    j0 += vl;
  }
}
#endif

template <typename PointT>
void
kernel<PointT>::fetchKernel(pcl::PointCloud<PointT>& kernel)
{
  switch (kernel_type_) {
  case SOBEL_X:
    sobelKernelX(kernel);
    break;
  case SOBEL_Y:
    sobelKernelY(kernel);
    break;
  case PREWITT_X:
    prewittKernelX(kernel);
    break;
  case PREWITT_Y:
    prewittKernelY(kernel);
    break;
  case ROBERTS_X:
    robertsKernelX(kernel);
    break;
  case ROBERTS_Y:
    robertsKernelY(kernel);
    break;
  case LOG:
    loGKernel(kernel);
    break;
  case DERIVATIVE_CENTRAL_X:
    derivativeXCentralKernel(kernel);
    break;
  case DERIVATIVE_FORWARD_X:
    derivativeXForwardKernel(kernel);
    break;
  case DERIVATIVE_BACKWARD_X:
    derivativeXBackwardKernel(kernel);
    break;
  case DERIVATIVE_CENTRAL_Y:
    derivativeYCentralKernel(kernel);
    break;
  case DERIVATIVE_FORWARD_Y:
    derivativeYForwardKernel(kernel);
    break;
  case DERIVATIVE_BACKWARD_Y:
    derivativeYBackwardKernel(kernel);
    break;
  case GAUSSIAN:
    gaussianKernel(kernel);
    break;
  }
}

template <typename PointT>
void
kernel<PointT>::gaussianKernel(pcl::PointCloud<PointT>& kernel)
{
  const int n = kernel_size_ * kernel_size_;
  kernel.resize(static_cast<std::size_t>(n));
  kernel.height = static_cast<std::uint32_t>(kernel_size_);
  kernel.width = static_cast<std::uint32_t>(kernel_size_);

  const float sigma_sqr = 2.f * sigma_ * sigma_;
  std::vector<float> buf(static_cast<std::size_t>(n));
  const std::size_t un = static_cast<std::size_t>(n);

#if defined(__RVV10__)
  computeGaussianKernelValuesRVV(buf, kernel_size_, sigma_sqr, n);
#else
  computeGaussianKernelValues(buf, kernel_size_, sigma_sqr, n);
#endif

#if defined(__RVV10__)
  float sum = sumReduceRVV(buf, un);
  normalizeAndWriteBackRVV(buf, sum, kernel, un);
#else
  float sum = 0;
  for (int idx = 0; idx < n; ++idx)
    sum += buf[static_cast<std::size_t>(idx)];

  // Normalizing: contiguous buffer so compiler can vectorize division
  for (int idx = 0; idx < n; ++idx)
    buf[static_cast<std::size_t>(idx)] /= sum;

  for (int idx = 0; idx < n; ++idx)
    kernel[static_cast<std::size_t>(idx)].intensity = buf[static_cast<std::size_t>(idx)];
#endif
}

template <typename PointT>
void
kernel<PointT>::loGKernel(pcl::PointCloud<PointT>& kernel)
{
  const int n = kernel_size_ * kernel_size_;
  kernel.resize(static_cast<std::size_t>(n));
  kernel.height = static_cast<std::uint32_t>(kernel_size_);
  kernel.width = static_cast<std::uint32_t>(kernel_size_);

  const float sigma_sqr = 2.f * sigma_ * sigma_;
  std::vector<float> buf(static_cast<std::size_t>(n));
  const std::size_t un = static_cast<std::size_t>(n);

#if defined(__RVV10__)
  computeLoGKernelValuesRVV(buf, kernel_size_, sigma_sqr, n);
#else
  computeLoGKernelValues(buf, kernel_size_, sigma_sqr, n);
#endif

#if defined(__RVV10__)
  float sum = sumReduceRVV(buf, un);
  normalizeAndWriteBackRVV(buf, sum, kernel, un);
#else
  float sum = 0;
  for (int idx = 0; idx < n; ++idx)
    sum += buf[static_cast<std::size_t>(idx)];

  // Normalizing: contiguous buffer so compiler can vectorize division
  for (int idx = 0; idx < n; ++idx)
    buf[static_cast<std::size_t>(idx)] /= sum;

  for (int idx = 0; idx < n; ++idx)
    kernel[static_cast<std::size_t>(idx)].intensity = buf[static_cast<std::size_t>(idx)];
#endif
}

template <typename PointT>
void
kernel<PointT>::sobelKernelX(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(9);
  kernel.height = 3;
  kernel.width = 3;
  kernel(0, 0).intensity = -1;
  kernel(1, 0).intensity = 0;
  kernel(2, 0).intensity = 1;
  kernel(0, 1).intensity = -2;
  kernel(1, 1).intensity = 0;
  kernel(2, 1).intensity = 2;
  kernel(0, 2).intensity = -1;
  kernel(1, 2).intensity = 0;
  kernel(2, 2).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::prewittKernelX(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(9);
  kernel.height = 3;
  kernel.width = 3;
  kernel(0, 0).intensity = -1;
  kernel(1, 0).intensity = 0;
  kernel(2, 0).intensity = 1;
  kernel(0, 1).intensity = -1;
  kernel(1, 1).intensity = 0;
  kernel(2, 1).intensity = 1;
  kernel(0, 2).intensity = -1;
  kernel(1, 2).intensity = 0;
  kernel(2, 2).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::robertsKernelX(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(4);
  kernel.height = 2;
  kernel.width = 2;
  kernel(0, 0).intensity = 1;
  kernel(1, 0).intensity = 0;
  kernel(0, 1).intensity = 0;
  kernel(1, 1).intensity = -1;
}

template <typename PointT>
void
kernel<PointT>::sobelKernelY(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(9);
  kernel.height = 3;
  kernel.width = 3;
  kernel(0, 0).intensity = -1;
  kernel(1, 0).intensity = -2;
  kernel(2, 0).intensity = -1;
  kernel(0, 1).intensity = 0;
  kernel(1, 1).intensity = 0;
  kernel(2, 1).intensity = 0;
  kernel(0, 2).intensity = 1;
  kernel(1, 2).intensity = 2;
  kernel(2, 2).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::prewittKernelY(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(9);
  kernel.height = 3;
  kernel.width = 3;
  kernel(0, 0).intensity = 1;
  kernel(1, 0).intensity = 1;
  kernel(2, 0).intensity = 1;
  kernel(0, 1).intensity = 0;
  kernel(1, 1).intensity = 0;
  kernel(2, 1).intensity = 0;
  kernel(0, 2).intensity = -1;
  kernel(1, 2).intensity = -1;
  kernel(2, 2).intensity = -1;
}

template <typename PointT>
void
kernel<PointT>::robertsKernelY(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(4);
  kernel.height = 2;
  kernel.width = 2;
  kernel(0, 0).intensity = 0;
  kernel(1, 0).intensity = 1;
  kernel(0, 1).intensity = -1;
  kernel(1, 1).intensity = 0;
}

template <typename PointT>
void
kernel<PointT>::derivativeXCentralKernel(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(3);
  kernel.height = 1;
  kernel.width = 3;
  kernel(0, 0).intensity = -1;
  kernel(1, 0).intensity = 0;
  kernel(2, 0).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::derivativeXForwardKernel(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(3);
  kernel.height = 1;
  kernel.width = 3;
  kernel(0, 0).intensity = 0;
  kernel(1, 0).intensity = -1;
  kernel(2, 0).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::derivativeXBackwardKernel(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(3);
  kernel.height = 1;
  kernel.width = 3;
  kernel(0, 0).intensity = -1;
  kernel(1, 0).intensity = 1;
  kernel(2, 0).intensity = 0;
}

template <typename PointT>
void
kernel<PointT>::derivativeYCentralKernel(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(3);
  kernel.height = 3;
  kernel.width = 1;
  kernel(0, 0).intensity = -1;
  kernel(0, 1).intensity = 0;
  kernel(0, 2).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::derivativeYForwardKernel(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(3);
  kernel.height = 3;
  kernel.width = 1;
  kernel(0, 0).intensity = 0;
  kernel(0, 1).intensity = -1;
  kernel(0, 2).intensity = 1;
}

template <typename PointT>
void
kernel<PointT>::derivativeYBackwardKernel(pcl::PointCloud<PointT>& kernel)
{
  kernel.resize(3);
  kernel.height = 3;
  kernel.width = 1;
  kernel(0, 0).intensity = -1;
  kernel(0, 1).intensity = 1;
  kernel(0, 2).intensity = 0;
}

template <typename PointT>
void
kernel<PointT>::setKernelType(KERNEL_ENUM kernel_type)
{
  kernel_type_ = kernel_type;
}

template <typename PointT>
void
kernel<PointT>::setKernelSize(int kernel_size)
{
  kernel_size_ = kernel_size;
}

template <typename PointT>
void
kernel<PointT>::setKernelSigma(float kernel_sigma)
{
  sigma_ = kernel_sigma;
}

} // namespace pcl
