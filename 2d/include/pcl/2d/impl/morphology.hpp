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

#include <pcl/2d/morphology.h>

#include <algorithm>
#include <limits>
#if defined(__RVV10__)
#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#endif

namespace pcl {

// Assumes input, kernel and output images have 0's and 1's only
template <typename PointT>
void
Morphology<PointT>::erosionBinary(pcl::PointCloud<PointT>& output)
{
  output.width = input_->width;
  output.height = input_->height;
  output.resize(input_->width * input_->height);
#if defined(__RVV10__)
  erosionBinaryRVV(output);
#else
  erosionBinaryStandard(output);
#endif
}

// Assumes input, kernel and output images have 0's and 1's only
template <typename PointT>
void
Morphology<PointT>::dilationBinary(pcl::PointCloud<PointT>& output)
{
  output.width = input_->width;
  output.height = input_->height;
  output.resize(input_->width * input_->height);
#if defined(__RVV10__)
  dilationBinaryRVV(output);
#else
  dilationBinaryStandard(output);
#endif
}

// Assumes input, kernel and output images have 0's and 1's only
template <typename PointT>
void
Morphology<PointT>::openingBinary(pcl::PointCloud<PointT>& output)
{
  PointCloudInPtr intermediate_output(new PointCloudIn);
  erosionBinary(*intermediate_output);
  this->setInputCloud(intermediate_output);
  dilationBinary(output);
}

// Assumes input, kernel and output images have 0's and 1's only
template <typename PointT>
void
Morphology<PointT>::closingBinary(pcl::PointCloud<PointT>& output)
{
  PointCloudInPtr intermediate_output(new PointCloudIn);
  dilationBinary(*intermediate_output);
  this->setInputCloud(intermediate_output);
  erosionBinary(output);
}

template <typename PointT>
void
Morphology<PointT>::erosionGray(pcl::PointCloud<PointT>& output)
{
  output.resize(input_->width * input_->height);
  output.width = input_->width;
  output.height = input_->height;
#if defined(__RVV10__)
  erosionGrayRVV(output);
#else
  erosionGrayStandard(output);
#endif
}

template <typename PointT>
void
Morphology<PointT>::dilationGray(pcl::PointCloud<PointT>& output)
{
  output.resize(input_->width * input_->height);
  output.width = input_->width;
  output.height = input_->height;
#if defined(__RVV10__)
  dilationGrayRVV(output);
  return;
#endif
  dilationGrayStandard(output);
}

template <typename PointT>
void
Morphology<PointT>::closingGray(pcl::PointCloud<PointT>& output)
{
  PointCloudInPtr intermediate_output(new PointCloudIn);
  dilationGray(*intermediate_output);
  this->setInputCloud(intermediate_output);
  erosionGray(output);
}

template <typename PointT>
void
Morphology<PointT>::openingGray(pcl::PointCloud<PointT>& output)
{
  PointCloudInPtr intermediate_output(new PointCloudIn);
  erosionGray(*intermediate_output);
  this->setInputCloud(intermediate_output);
  dilationGray(output);
}

#if defined(__RVV10__)
template <typename PointT>
void
Morphology<PointT>::erosionBinaryRVV(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(structuring_element_->width);
  const int kh = static_cast<int>(structuring_element_->height);
  const int kh_half = kh / 2;
  const int kw_half = kw / 2;

  const std::size_t point_stride = sizeof(PointT);
  const std::size_t intensity_offset = offsetof(PointT, intensity);
  const std::uint8_t* input_base =
      reinterpret_cast<const std::uint8_t*>(input_->points.data());
  std::uint8_t* output_base = reinterpret_cast<std::uint8_t*>(output.points.data());

  const int row_lo = kh_half;
  const int row_hi = ih - kh_half;
  const int col_lo = kw_half;
  const int col_hi = iw - kw_half;

  // --- Center (safe) region: min over kernel-1 neighbors, then output = (center==1 &&
  // min==1) ---
  if (row_hi > row_lo && col_hi > col_lo) {
    const float v_one = 1.0f;
    const float v_zero = 0.0f;
    for (int i = row_lo; i < row_hi; ++i) {
      int j0 = col_lo;
      while (j0 < col_hi) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(col_hi - j0));
        vfloat32m2_t v_min =
            __riscv_vfmv_v_f_f32m2(std::numeric_limits<float>::max(), vl);

        for (int k = 0; k < kh; ++k) {
          const int base_row = i + k - kh_half;
          const std::size_t row_offset = static_cast<std::size_t>(base_row) * iw;
          for (int l = 0; l < kw; ++l) {
            if ((*structuring_element_)(l, k).intensity == 0.0f)
              continue;
            const int base_col = j0 + l - kw_half;
            const float* in_ptr = reinterpret_cast<const float*>(
                input_base +
                (row_offset + static_cast<std::size_t>(base_col)) * point_stride +
                intensity_offset);
            vfloat32m2_t v_in = __riscv_vlse32_v_f32m2(in_ptr, point_stride, vl);
            v_min = __riscv_vfmin_vv_f32m2(v_min, v_in, vl);
          }
        }

        const float* center_ptr = reinterpret_cast<const float*>(
            input_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride +
            intensity_offset);
        vfloat32m2_t v_center = __riscv_vlse32_v_f32m2(center_ptr, point_stride, vl);
        vbool16_t m_center_one = __riscv_vmfeq_vf_f32m2_b16(v_center, v_one, vl);
        vbool16_t m_min_one = __riscv_vmfeq_vf_f32m2_b16(v_min, v_one, vl);
        vbool16_t mask = __riscv_vmand_mm_b16(m_center_one, m_min_one, vl);

        vfloat32m2_t v_zero_vec = __riscv_vfmv_v_f_f32m2(v_zero, vl);
        vfloat32m2_t v_out = __riscv_vfmerge_vfm_f32m2(v_zero_vec, v_one, mask, vl);

        float* out_ptr = reinterpret_cast<float*>(
            output_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride +
            intensity_offset);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_out, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }

  // --- Edge region: single-pixel helper (same semantics as standard) ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelErosionBinary(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelErosionBinary(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelErosionBinary(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelErosionBinary(i, j, output);
  }
}
#endif

template <typename PointT>
void
Morphology<PointT>::erosionBinaryStandard(pcl::PointCloud<PointT>& output)
{
  const int height = input_->height;
  const int width = input_->width;
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      computePixelErosionBinary(i, j, output);
}

template <typename PointT>
void
Morphology<PointT>::computePixelErosionBinary(int i,
                                              int j,
                                              pcl::PointCloud<PointT>& output) const
{
  const int height = input_->height;
  const int width = input_->width;
  const int kernel_height = structuring_element_->height;
  const int kernel_width = structuring_element_->width;
  const int kh_half = kernel_height / 2;
  const int kw_half = kernel_width / 2;

  if ((*input_)(j, i).intensity == 0.0f) {
    output(j, i).intensity = 0.0f;
    return;
  }

  for (int k = 0; k < kernel_height; ++k) {
    const int input_row = i + k - kh_half;
    if (input_row < 0 || input_row >= height)
      continue;

    for (int l = 0; l < kernel_width; ++l) {
      if ((*structuring_element_)(l, k).intensity == 0.0f)
        continue;

      const int input_col = j + l - kw_half;
      if (input_col < 0 || input_col >= width)
        continue;

      if ((*input_)(input_col, input_row).intensity != 1.0f) {
        output(j, i).intensity = 0.0f;
        return;
      }
    }
  }
  output(j, i).intensity = 1.0f;
}

#if defined(__RVV10__)
template <typename PointT>
void
Morphology<PointT>::dilationBinaryRVV(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(structuring_element_->width);
  const int kh = static_cast<int>(structuring_element_->height);
  const int kh_half = kh / 2;
  const int kw_half = kw / 2;

  const std::size_t point_stride = sizeof(PointT);
  const std::size_t intensity_offset = offsetof(PointT, intensity);
  const std::uint8_t* input_base =
      reinterpret_cast<const std::uint8_t*>(input_->points.data());
  std::uint8_t* output_base = reinterpret_cast<std::uint8_t*>(output.points.data());

  const int row_lo = kh_half;
  const int row_hi = ih - kh_half;
  const int col_lo = kw_half;
  const int col_hi = iw - kw_half;

  // --- Center (safe) region: max over kernel-1 neighbors, then output = (max==1) ---
  if (row_hi > row_lo && col_hi > col_lo) {
    const float v_one = 1.0f;
    const float v_zero = 0.0f;
    for (int i = row_lo; i < row_hi; ++i) {
      int j0 = col_lo;
      while (j0 < col_hi) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(col_hi - j0));
        vfloat32m2_t v_max = __riscv_vfmv_v_f_f32m2(-1.0f, vl);

        for (int k = 0; k < kh; ++k) {
          const int base_row = i + k - kh_half;
          const std::size_t row_offset = static_cast<std::size_t>(base_row) * iw;
          for (int l = 0; l < kw; ++l) {
            if ((*structuring_element_)(l, k).intensity == 0.0f)
              continue;
            const int base_col = j0 + l - kw_half;
            const float* in_ptr = reinterpret_cast<const float*>(
                input_base +
                (row_offset + static_cast<std::size_t>(base_col)) * point_stride +
                intensity_offset);
            vfloat32m2_t v_in = __riscv_vlse32_v_f32m2(in_ptr, point_stride, vl);
            v_max = __riscv_vfmax_vv_f32m2(v_max, v_in, vl);
          }
        }

        vbool16_t mask = __riscv_vmfeq_vf_f32m2_b16(v_max, v_one, vl);
        vfloat32m2_t v_zero_vec = __riscv_vfmv_v_f_f32m2(v_zero, vl);
        vfloat32m2_t v_out = __riscv_vfmerge_vfm_f32m2(v_zero_vec, v_one, mask, vl);

        float* out_ptr = reinterpret_cast<float*>(
            output_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride +
            intensity_offset);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_out, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }

  // --- Edge region: single-pixel helper (same semantics as standard) ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelDilationBinary(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelDilationBinary(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelDilationBinary(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelDilationBinary(i, j, output);
  }
}
#endif

template <typename PointT>
void
Morphology<PointT>::dilationBinaryStandard(pcl::PointCloud<PointT>& output)
{
  const int height = input_->height;
  const int width = input_->width;
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      computePixelDilationBinary(i, j, output);
}

template <typename PointT>
void
Morphology<PointT>::computePixelDilationBinary(int i,
                                               int j,
                                               pcl::PointCloud<PointT>& output) const
{
  const int height = input_->height;
  const int width = input_->width;
  const int kernel_height = structuring_element_->height;
  const int kernel_width = structuring_element_->width;
  const int kh_half = kernel_height / 2;
  const int kw_half = kernel_width / 2;

  for (int k = 0; k < kernel_height; ++k) {
    const int input_row = i + k - kh_half;
    if (input_row < 0 || input_row >= height)
      continue;

    for (int l = 0; l < kernel_width; ++l) {
      if ((*structuring_element_)(l, k).intensity == 0.0f)
        continue;

      const int input_col = j + l - kw_half;
      if (input_col < 0 || input_col >= width)
        continue;

      if ((*input_)(input_col, input_row).intensity == 1.0f) {
        output(j, i).intensity = 1.0f;
        return;
      }
    }
  }
  output(j, i).intensity = 0.0f;
}

#ifdef __RVV10__
template <typename PointT>
void
Morphology<PointT>::erosionGrayRVV(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(structuring_element_->width);
  const int kh = static_cast<int>(structuring_element_->height);
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
        vfloat32m2_t v_min =
            __riscv_vfmv_v_f_f32m2(std::numeric_limits<float>::max(), vl);

        for (int k = 0; k < kh; ++k) {
          const int base_row = i + k - kh_half;
          const std::size_t row_offset = static_cast<std::size_t>(base_row) * iw;
          for (int l = 0; l < kw; ++l) {
            // We only check for 1's in the kernel
            if ((*structuring_element_)(l, k).intensity == 0.0f)
              continue;

            const int base_col = j0 + l - kw_half;
            const float* in_ptr = reinterpret_cast<const float*>(
                input_base +
                (row_offset + static_cast<std::size_t>(base_col)) * point_stride +
                intensity_offset);
            vfloat32m2_t v_in = __riscv_vlse32_v_f32m2(in_ptr, point_stride, vl);
            v_min = __riscv_vfmin_vv_f32m2(v_min, v_in, vl);
          }
        }

        float* out_ptr = reinterpret_cast<float*>(
            output_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride +
            intensity_offset);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_min, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }

  // --- Edge region: 4 explicit loops, no branch on center pixels ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelMin(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelMin(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelMin(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelMin(i, j, output);
  }
}
#endif

template <typename PointT>
void
Morphology<PointT>::erosionGrayStandard(pcl::PointCloud<PointT>& output)
{
  const int height = input_->height;
  const int width = input_->width;
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      computePixelMin(i, j, output);
}

template <typename PointT>
void
Morphology<PointT>::computePixelMin(int i, int j, pcl::PointCloud<PointT>& output) const
{
  const int height = input_->height;
  const int width = input_->width;
  const int kernel_height = structuring_element_->height;
  const int kernel_width = structuring_element_->width;
  const int kh_half = kernel_height / 2;
  const int kw_half = kernel_width / 2;
  float min_val = (std::numeric_limits<float>::max)();
  bool found = false;

  for (int k = 0; k < kernel_height; ++k) {
    const int input_row = i + k - kh_half;
    if (input_row < 0 || input_row >= height)
      continue;

    for (int l = 0; l < kernel_width; ++l) {
      // We only check for 1's in the kernel
      if ((*structuring_element_)(l, k).intensity == 0.0f)
        continue;

      const int input_col = j + l - kw_half;
      if (input_col < 0 || input_col >= width)
        continue;

      found = true;
      min_val = std::min(min_val, (*input_)(input_col, input_row).intensity);
    }
  }
  output(j, i).intensity = found ? min_val : -1.0f;
}

#if defined(__RVV10__)
template <typename PointT>
void
Morphology<PointT>::dilationGrayRVV(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(structuring_element_->width);
  const int kh = static_cast<int>(structuring_element_->height);
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
        vfloat32m2_t v_max =
            __riscv_vfmv_v_f_f32m2(std::numeric_limits<float>::lowest(), vl);

        for (int k = 0; k < kh; ++k) {
          const int base_row = i + k - kh_half;
          const std::size_t row_offset = static_cast<std::size_t>(base_row) * iw;
          for (int l = 0; l < kw; ++l) {
            // We only check for 1's in the kernel
            if ((*structuring_element_)(l, k).intensity == 0.0f)
              continue;

            const int base_col = j0 + l - kw_half;
            const float* in_ptr = reinterpret_cast<const float*>(
                input_base +
                (row_offset + static_cast<std::size_t>(base_col)) * point_stride +
                intensity_offset);
            vfloat32m2_t v_in = __riscv_vlse32_v_f32m2(in_ptr, point_stride, vl);
            v_max = __riscv_vfmax_vv_f32m2(v_max, v_in, vl);
          }
        }

        float* out_ptr = reinterpret_cast<float*>(
            output_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride +
            intensity_offset);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_max, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }

  // --- Edge region: 4 explicit loops, no branch on center pixels ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelMax(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelMax(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelMax(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelMax(i, j, output);
  }
}
#endif

template <typename PointT>
void
Morphology<PointT>::dilationGrayStandard(pcl::PointCloud<PointT>& output)
{
  const int height = input_->height;
  const int width = input_->width;
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      computePixelMax(i, j, output);
}

template <typename PointT>
void
Morphology<PointT>::computePixelMax(int i, int j, pcl::PointCloud<PointT>& output) const
{
  const int height = input_->height;
  const int width = input_->width;
  const int kernel_height = structuring_element_->height;
  const int kernel_width = structuring_element_->width;
  const int kh_half = kernel_height / 2;
  const int kw_half = kernel_width / 2;
  float max_val = (std::numeric_limits<float>::lowest)();
  bool found = false;

  for (int k = 0; k < kernel_height; ++k) {
    const int input_row = i + k - kh_half;
    if (input_row < 0 || input_row >= height)
      continue;

    for (int l = 0; l < kernel_width; ++l) {
      // We only check for 1's in the kernel
      if ((*structuring_element_)(l, k).intensity == 0.0f)
        continue;

      const int input_col = j + l - kw_half;
      if (input_col < 0 || input_col >= width)
        continue;

      found = true;
      max_val = std::max(max_val, (*input_)(input_col, input_row).intensity);
    }
  }
  output(j, i).intensity = found ? max_val : -1.0f;
}

template <typename PointT>
void
Morphology<PointT>::subtractionBinary(pcl::PointCloud<PointT>& output,
                                      const pcl::PointCloud<PointT>& input1,
                                      const pcl::PointCloud<PointT>& input2)
{
  const int height = (input1.height < input2.height) ? input1.height : input2.height;
  const int width = (input1.width < input2.width) ? input1.width : input2.width;
  output.width = width;
  output.height = height;
  // RVV10 手动向量化 set operation 在 Milkv-Jupiter 上实测显著慢于标量。
  // 原因主要来自 AoS 结构中仅处理 `intensity` 字段导致的 strided load/store，
  // 以及 `vbool16` 掩码与 `vmerge` 链路带来的指令/访存开销未被足够摊薄。
  // 因此这里保留标量实现，避免在该数据规模与平台配置下产生回退。
  output.resize(height * width);

  for (std::size_t i = 0; i < output.size(); ++i) {
    if (input1[i].intensity == 1.0f && input2[i].intensity == 0.0f)
      output[i].intensity = 1.0f;
    else
      output[i].intensity = 0.0f;
  }
}

template <typename PointT>
void
Morphology<PointT>::unionBinary(pcl::PointCloud<PointT>& output,
                                const pcl::PointCloud<PointT>& input1,
                                const pcl::PointCloud<PointT>& input2)
{
  const int height = (input1.height < input2.height) ? input1.height : input2.height;
  const int width = (input1.width < input2.width) ? input1.width : input2.width;
  output.width = width;
  output.height = height;
  // 同 `subtractionBinary`：保留标量 set operation，避免 RVV10
  // 手写向量化在该场景下回退。
  output.resize(height * width);

  for (std::size_t i = 0; i < output.size(); ++i) {
    if (input1[i].intensity == 1.0f || input2[i].intensity == 1.0f)
      output[i].intensity = 1.0f;
    else
      output[i].intensity = 0.0f;
  }
}

template <typename PointT>
void
Morphology<PointT>::intersectionBinary(pcl::PointCloud<PointT>& output,
                                       const pcl::PointCloud<PointT>& input1,
                                       const pcl::PointCloud<PointT>& input2)
{
  const int height = (input1.height < input2.height) ? input1.height : input2.height;
  const int width = (input1.width < input2.width) ? input1.width : input2.width;
  output.width = width;
  output.height = height;
  // 同 `subtractionBinary`：保留标量 set operation，避免 RVV10
  // 手写向量化在该场景下回退。
  output.resize(height * width);

  for (std::size_t i = 0; i < output.size(); ++i) {
    if (input1[i].intensity == 1.0f && input2[i].intensity == 1.0f)
      output[i].intensity = 1.0f;
    else
      output[i].intensity = 0.0f;
  }
}

template <typename PointT>
void
Morphology<PointT>::structuringElementCircular(pcl::PointCloud<PointT>& kernel,
                                               const int radius)
{
  const int dim = 2 * radius;
  kernel.height = dim;
  kernel.width = dim;
  kernel.resize(dim * dim);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (((i - radius) * (i - radius) + (j - radius) * (j - radius)) < radius * radius)
        kernel(j, i).intensity = 1;
      else
        kernel(j, i).intensity = 0;
    }
  }
}

template <typename PointT>
void
Morphology<PointT>::structuringElementRectangle(pcl::PointCloud<PointT>& kernel,
                                                const int height,
                                                const int width)
{
  kernel.height = height;
  kernel.width = width;
  kernel.resize(height * width);
  for (std::size_t i = 0; i < kernel.size(); ++i)
    kernel[i].intensity = 1;
}

template <typename PointT>
void
Morphology<PointT>::setStructuringElement(const PointCloudInPtr& structuring_element)
{
  structuring_element_ = structuring_element;
}

} // namespace pcl
