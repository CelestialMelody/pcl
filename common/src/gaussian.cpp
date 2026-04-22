/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, wwww.pointclouds.org
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

#include <pcl/common/gaussian.h>
#include <cassert>

#if defined(__RVV10__)
#include <riscv_vector.h>

#include <cstddef>
#endif

/*
 * PointCloud<float> convolveRows / convolveCols: where this code lives and how RVV is chosen
 *
 * Location:
 *   This translation unit is linked into libpcl_common. Templated convolve overloads
 *   (PointT with std::function field access) are in common/include/pcl/common/impl/gaussian.hpp
 *   and are not built from this file.
 *
 * Compile-time dispatch:
 *   The member functions call convolveRowsRVV or convolveRowsStandard depending on whether
 *   __RVV10__ was defined when this .cpp was compiled into libpcl_common. Macros on the
 *   application, unit test, or benchmark executable do not change machine code inside an
 *   already installed shared object. Changing only the test program flags without rebuilding
 *   and reinstalling the library leaves the convolution path unchanged.
 *
 * After edits:
 *   Rebuild and install pcl_common, then relink benchmarks and tests, so timings and behavior
 *   match the new object code.
 *
 * Scalar versus RVV benchmarks:
 *   Use two libpcl_common builds (for example one with __RVV10__ for this TU and one without)
 *   and switch install prefix or LD_LIBRARY_PATH.
 *
 * float 卷积只在本文件实现；是否 RVV 取决于把本 TU 编进 libpcl_common 时是否定义
 * __RVV10__，与可执行文件自己的宏无关。改代码后需重装库。标量与 RVV 对比需要单独编译两套库。
 */
namespace {

#if defined(__RVV10__)
/** \brief RVV path: horizontal separable convolution along rows (1D kernel in x). */
void
convolveRowsRVV (const pcl::PointCloud<float> &input,
                 pcl::PointCloud<float> &output,
                 const Eigen::VectorXf &kernel,
                 std::size_t kernel_width,
                 std::size_t radius)
{
  const std::size_t width = input.width;
  const std::size_t height = input.height;
  const float *const in_p = input.data ();
  float *const out_p = output.data ();
  const float *const kptr = kernel.data ();
  const int kw = static_cast<int> (kernel_width);

  for (std::size_t j = 0; j < height; ++j)
  {
    const float *const in_row = in_p + j * width;
    float *const out_row = out_p + j * width;

    std::size_t i = 0;
    for (; i < radius; ++i)
      out_row[i] = 0.f;

    const std::size_t i_end = width - radius;
    while (i < i_end)
    {
      const std::size_t vl = __riscv_vsetvl_e32m2 (i_end - i);
      /* Next strip window base: (i+vl)-radius; hints cache for the following output chunk. */
      __builtin_prefetch (in_row + (i + vl) - radius, 0, 3);

      const float *const win_base = in_row + (i - radius);
      vfloat32m2_t vin = __riscv_vle32_v_f32m2 (win_base, vl);
      vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2 (0.f, vl);
      acc = __riscv_vfmacc_vf_f32m2 (acc, kptr[kw], vin, vl);
      for (int s = 1; s <= kw; ++s)
      {
        const float tail = win_base[s + vl - 1];
        vin = __riscv_vfslide1down_vf_f32m2 (vin, tail, vl);
        acc = __riscv_vfmacc_vf_f32m2 (acc, kptr[kw - s], vin, vl);
      }
      __riscv_vse32_v_f32m2 (out_row + i, acc, vl);
      i += vl;
    }

    for (; i < width; ++i)
      out_row[i] = 0.f;
  }
}

/** \brief RVV path: vertical separable convolution along columns (1D kernel in y). */
void
convolveColsRVV (const pcl::PointCloud<float> &input,
                 pcl::PointCloud<float> &output,
                 const Eigen::VectorXf &kernel,
                 std::size_t kernel_width,
                 std::size_t radius)
{
  const std::size_t width = input.width;
  const std::size_t height = input.height;
  const float *const in_p = input.data ();
  float *const out_p = output.data ();
  const float *const kptr = kernel.data ();
  const int kw = static_cast<int> (kernel_width);
  const ptrdiff_t row_stride_bytes =
      static_cast<ptrdiff_t> (width) * static_cast<ptrdiff_t> (sizeof (float));

  for (std::size_t i = 0; i < width; ++i)
  {
    std::size_t j = 0;
    for (; j < radius; ++j)
      out_p[i + j * width] = 0.f;

    const std::size_t j_end = height - radius;
    while (j < j_end)
    {
      const std::size_t vl = __riscv_vsetvl_e32m2 (j_end - j);
      /* Strided column strip: taps are not a contiguous slide; keep one vlse per tap. */
      __builtin_prefetch (in_p + i + (j + vl) * width, 0, 3);

      vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2 (0.f, vl);
      for (int s = 0; s <= kw; ++s)
      {
        const float *const src =
            in_p + i + (j - radius + static_cast<std::size_t> (s)) * width;
        const vfloat32m2_t vin = __riscv_vlse32_v_f32m2 (src, row_stride_bytes, vl);
        acc = __riscv_vfmacc_vf_f32m2 (acc, kptr[kw - s], vin, vl);
      }
      __riscv_vsse32_v_f32m2 (out_p + i + j * width, row_stride_bytes, acc, vl);
      j += vl;
    }

    for (; j < height; ++j)
      out_p[i + j * width] = 0.f;
  }
}
#endif // defined(__RVV10__)

/** \brief Scalar path: horizontal separable convolution along rows (1D kernel in x). */
void
convolveRowsStandard (const pcl::PointCloud<float> &input,
                      pcl::PointCloud<float> &output,
                      const Eigen::VectorXf &kernel,
                      std::size_t kernel_width,
                      std::size_t radius)
{
  for (std::size_t j = 0; j < input.height; j++)
  {
    std::size_t i = 0;
    for (; i < radius; i++)
      output (i, j) = 0;

    for (; i < input.width - radius; i++)
    {
      output (i, j) = 0;
      for (int k = static_cast<int> (kernel_width), l = static_cast<int> (i - radius);
           k >= 0; k--, l++)
        output (i, j) += input (l, j) * kernel[k];
    }

    for (; i < input.width; i++)
      output (i, j) = 0;
  }
}

/** \brief Scalar path: vertical separable convolution along columns (1D kernel in y). */
void
convolveColsStandard (const pcl::PointCloud<float> &input,
                      pcl::PointCloud<float> &output,
                      const Eigen::VectorXf &kernel,
                      std::size_t kernel_width,
                      std::size_t radius)
{
  for (std::size_t i = 0; i < input.width; i++)
  {
    std::size_t j = 0;
    for (; j < radius; j++)
      output (i, j) = 0;

    for (; j < input.height - radius; j++)
    {
      output (i, j) = 0;
      for (int k = static_cast<int> (kernel_width), l = static_cast<int> (j - radius);
           k >= 0; k--, l++)
        output (i, j) += input (i, l) * kernel[k];
    }

    for (; j < input.height; j++)
      output (i, j) = 0;
  }
}

} // namespace

void
pcl::GaussianKernel::compute (float sigma,
                              Eigen::VectorXf &kernel,
                              unsigned kernel_width) const
{
  assert (kernel_width %2 == 1);
  assert (sigma >= 0);
  kernel.resize (kernel_width);
  static const float factor = 0.01f;
  static const float max_gauss = 1.0f;
  const int hw = kernel_width / 2;
  float sigma_sqr = 1.0f / (2.0f * sigma * sigma);
  for (int i = -hw, j = 0, k = kernel_width - 1; i < 0 ; i++, j++, k--)
    kernel[k] = kernel[j] = std::exp (-static_cast<float>(i) * static_cast<float>(i) * sigma_sqr);
  kernel[hw] = 1;
  unsigned g_width = kernel_width;
  for (unsigned i = 0; std::fabs (kernel[i]/max_gauss) < factor; i++, g_width-= 2) ;
  if (g_width == kernel_width)
  {
    PCL_THROW_EXCEPTION (pcl::KernelWidthTooSmallException,
                        "kernel width " << kernel_width
                        << "is too small for the given sigma " << sigma);
    return;
  }

  // Shift and resize if width less than kernel_width
  unsigned shift = (kernel_width - g_width)/2;
  for (unsigned i =0; i < g_width; i++)
    kernel[i] = kernel[i + shift];
  kernel.conservativeResize (g_width);

  // Normalize
  kernel/= kernel.sum ();
}

void
pcl::GaussianKernel::compute (float sigma,
                              Eigen::VectorXf &kernel,
                              Eigen::VectorXf &derivative,
                              unsigned kernel_width) const
{
  assert (kernel_width %2 == 1);
  assert (sigma >= 0);
  kernel.resize (kernel_width);
  derivative.resize (kernel_width);
  const float factor = 0.01f;
  float max_gauss = 1.0f, max_deriv = static_cast<float>(sigma * std::exp (-0.5));
  int hw = kernel_width / 2;

  float sigma_sqr = 1.0f / (2.0f * sigma * sigma);
  for (int i = -hw, j = 0, k = kernel_width - 1; i < 0 ; i++, j++, k--)
  {
    kernel[k] = kernel[j] = std::exp (-static_cast<float>(i) * static_cast<float>(i) * sigma_sqr);
    derivative[j] = -static_cast<float>(i) * kernel[j];
    derivative[k] = -derivative[j];
  }
  kernel[hw] = 1;
  derivative[hw] = 0;
  // Compute kernel and derivative true width
  unsigned g_width = kernel_width;
  unsigned d_width = kernel_width;
  for (unsigned i = 0; std::fabs (derivative[i]/max_deriv) < factor; i++, d_width-= 2) ;
  for (unsigned i = 0; std::fabs (kernel[i]/max_gauss) < factor; i++, g_width-= 2) ;
  if (g_width == kernel_width || d_width == kernel_width)
  {
    PCL_THROW_EXCEPTION (KernelWidthTooSmallException,
                        "kernel width " << kernel_width
                        << "is too small for the given sigma " << sigma);
    return;
  }

  // Shift and resize if width less than kernel_width
  // Kernel
  unsigned shift = (kernel_width - g_width)/2;
  for (unsigned i =0; i < g_width; i++)
    kernel[i] = kernel[i + shift];
  // Normalize kernel
  kernel.conservativeResize (g_width);
  kernel/= kernel.sum ();

  // Derivative
  shift = (kernel_width - d_width)/2;
  for (unsigned i =0; i < d_width; i++)
    derivative[i] = derivative[i + shift];
  derivative.conservativeResize (d_width);
  // Normalize derivative
  hw = d_width / 2;
  float den = 0;
  for (int i = -hw ; i <= hw ; i++)
    den -=  static_cast<float>(i) * derivative[i+hw];
  derivative/= den;
}

void
pcl::GaussianKernel::convolveRows (const pcl::PointCloud<float>& input,
                                   const Eigen::VectorXf& kernel,
                                   pcl::PointCloud<float>& output) const
{
  assert (kernel.size () % 2 == 1);
  std::size_t kernel_width = kernel.size () -1;
  std::size_t radius = kernel.size () / 2;
  pcl::PointCloud<float> copied_input;
  const pcl::PointCloud<float>* unaliased_input;
  if (&input != &output)
  {
    if (output.height < input.height || output.width < input.width)
    {
      output.resize (static_cast<uindex_t>(input.width), static_cast<uindex_t>(input.height)); // Casting is necessary to resolve ambiguous call to resize
    }
    unaliased_input = &input;
  }
  else
  {
    copied_input = input;
    unaliased_input = &copied_input;
  }

  /* Chosen when libpcl_common was built (__RVV10__ on this TU). See file header. 中文：见文件头。 */
#if defined(__RVV10__)
  convolveRowsRVV (*unaliased_input, output, kernel, kernel_width, radius);
#else
  convolveRowsStandard (*unaliased_input, output, kernel, kernel_width, radius);
#endif
}

void
pcl::GaussianKernel::convolveCols (const pcl::PointCloud<float>& input,
                                   const Eigen::VectorXf& kernel,
                                   pcl::PointCloud<float>& output) const
{
  assert (kernel.size () % 2 == 1);
  std::size_t kernel_width = kernel.size () -1;
  std::size_t radius = kernel.size () / 2;
  pcl::PointCloud<float> copied_input;
  const pcl::PointCloud<float>* unaliased_input;
  if (&input != &output)
  {
    if (output.height < input.height || output.width < input.width)
    {
      output.resize (static_cast<uindex_t>(input.width), static_cast<uindex_t>(input.height)); // Casting is necessary to resolve ambiguous call to resize
    }
    unaliased_input = &input;
  }
  else
  {
    copied_input = input;
    unaliased_input = &copied_input;
  }

  /* Chosen when libpcl_common was built (__RVV10__ on this TU). See file header. 中文：见文件头。 */
#if defined(__RVV10__)
  convolveColsRVV (*unaliased_input, output, kernel, kernel_width, radius);
#else
  convolveColsStandard (*unaliased_input, output, kernel, kernel_width, radius);
#endif
}
