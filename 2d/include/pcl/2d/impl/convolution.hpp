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

#include <pcl/2d/convolution.h>
#if defined(__RVV10__)
#include <riscv_vector.h>
#include <cstddef>
#endif

namespace pcl {
template <typename PointT>
void
Convolution<PointT>::filter(pcl::PointCloud<PointT>& output)
{
  output = *input_;
#if defined(__RVV10__)
  filterRVV(output);
#else
  filterStandard(output);
#endif
}

template <typename PointT>
void
Convolution<PointT>::computePixelIntensity(int i, int j, pcl::PointCloud<PointT>& output) const
{
  int input_row = 0;
  int input_col = 0;
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(kernel_.width);
  const int kh = static_cast<int>(kernel_.height);
  const int kh_half = kh / 2;
  const int kw_half = kw / 2;
  float intensity = 0.0f;

  switch (boundary_options_) {
  default:
  case BOUNDARY_OPTION_CLAMP:
    for (int k = 0; k < kh; ++k) {
      const int ikkh = i + k - kh_half;
      if (ikkh < 0)
        input_row = 0;
      else if (ikkh >= ih)
        input_row = ih - 1;
      else
        input_row = ikkh;

      for (int l = 0; l < kw; ++l) {
        const int jlkw = j + l - kw_half;
        if (jlkw < 0)
          input_col = 0;
        else if (jlkw >= iw)
          input_col = iw - 1;
        else
          input_col = jlkw;

        intensity += kernel_(l, k).intensity * (*input_)(input_col, input_row).intensity;
      }
    }
    break;

  case BOUNDARY_OPTION_MIRROR:
    for (int k = 0; k < kh; ++k) {
      const int ikkh = i + k - kh_half;
      if (ikkh < 0)
        input_row = -ikkh - 1;
      else if (ikkh >= ih)
        input_row = 2 * ih - 1 - ikkh;
      else
        input_row = ikkh;

      for (int l = 0; l < kw; ++l) {
        const int jlkw = j + l - kw_half;
        if (jlkw < 0)
          input_col = -jlkw - 1;
        else if (jlkw >= iw)
          input_col = 2 * iw - 1 - jlkw;
        else
          input_col = jlkw;

        intensity += kernel_(l, k).intensity * (*input_)(input_col, input_row).intensity;
      }
    }
    break;

  case BOUNDARY_OPTION_ZERO_PADDING:
    for (int k = 0; k < kh; ++k) {
      const int ikkh = i + k - kh_half;
      if (ikkh < 0 || ikkh >= ih)
        continue;

      for (int l = 0; l < kw; ++l) {
        const int jlkw = j + l - kw_half;
        if (jlkw < 0 || jlkw >= iw)
          continue;
        intensity += kernel_(l, k).intensity * (*input_)(jlkw, ikkh).intensity;
      }
    }
    break;
  }
  output(j, i).intensity = intensity;
}

template <typename PointT>
void
Convolution<PointT>::filterStandard(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  for (int i = 0; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelIntensity(i, j, output);
}

#if defined(__RVV10__)
template <typename PointT>
void
Convolution<PointT>::filterRVV(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(kernel_.width);
  const int kh = static_cast<int>(kernel_.height);
  const int kh_half = kh / 2;
  const int kw_half = kw / 2;

  const std::size_t point_stride = sizeof(PointT);
  const std::size_t intensity_offset = offsetof(PointT, intensity);
  const std::uint8_t* input_base =
      reinterpret_cast<const std::uint8_t*>(input_->points.data());
  std::uint8_t* output_base = reinterpret_cast<std::uint8_t*>(output.points.data());

  // --- Center (safe) region: kernel fully inside image, vectorized ---
  const int row_lo = kh_half;
  const int row_hi = ih - kh_half;
  const int col_lo = kw_half;
  const int col_hi = iw - kw_half;

  if (row_hi > row_lo && col_hi > col_lo) {
    for (int i = row_lo; i < row_hi; ++i) {
      int j0 = col_lo;
      while (j0 < col_hi) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(col_hi - j0));
        vfloat32m2_t v_acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int k = 0; k < kh; ++k) {
          const int base_row = i + k - kh_half;
          const std::size_t row_offset = static_cast<std::size_t>(base_row) * iw;
          const std::size_t in_start_col = static_cast<std::size_t>(j0 - kw_half);
          const std::uint8_t* in_ptr_byte_base =
              input_base + (row_offset + in_start_col) * point_stride + intensity_offset;
          for (int l = 0; l < kw; ++l) {
            const float kernel_val = kernel_(l, k).intensity;
            const float* in_ptr = reinterpret_cast<const float*>(in_ptr_byte_base);
            vfloat32m2_t v_in = __riscv_vlse32_v_f32m2(in_ptr, point_stride, vl);
            v_acc = __riscv_vfmacc_vf_f32m2(v_acc, kernel_val, v_in, vl);
            in_ptr_byte_base += point_stride; // next l => next input column (+1)
          }
        }

        float* out_ptr = reinterpret_cast<float*>(
            output_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride + intensity_offset);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_acc, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }

  // --- Edge region: 4 explicit loops, no branch on center pixels ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelIntensity(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelIntensity(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelIntensity(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelIntensity(i, j, output);
  }
}
#endif

} // namespace pcl
