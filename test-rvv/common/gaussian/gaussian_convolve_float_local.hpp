/*
 * 与 **common/src/gaussian.cpp** 中匿名命名空间内 float 可分离行/列卷积算法**保持一致**的副本
 *（仅接口不同：此处为行主序 float* + 宽/高）。不依赖 PCL/PointCloud，供本目录在**不链 libpcl_common** 时对拍。
 * 若修改 gaussian.cpp 中卷积实现，请同步更新此处。
 */
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__RVV10__)
#include <riscv_vector.h>
#endif

namespace gaussian_convolve_local
{

/** nkernel 为一维核系数个数（奇数）；内部分别使用 kernel_width=nkernel-1、radius=nkernel/2。 */
inline void
convolveRowsStandard (const float *in_p, float *out_p, std::size_t width, std::size_t height,
                      const float *kptr, std::size_t nkernel)
{
  const int kw = static_cast<int> (nkernel) - 1;
  const std::size_t radius = nkernel / 2;

  for (std::size_t j = 0; j < height; j++)
  {
    std::size_t i = 0;
    for (; i < radius; i++)
      out_p[i + j * width] = 0;

    for (; i < width - radius; i++)
    {
      out_p[i + j * width] = 0;
      for (int k = kw, l = static_cast<int> (i - radius); k >= 0; k--, l++)
        out_p[i + j * width] += in_p[l + j * width] * kptr[k];
    }

    for (; i < width; i++)
      out_p[i + j * width] = 0;
  }
}

inline void
convolveColsStandard (const float *in_p, float *out_p, std::size_t width, std::size_t height,
                      const float *kptr, std::size_t nkernel)
{
  const int kw = static_cast<int> (nkernel) - 1;
  const std::size_t radius = nkernel / 2;

  for (std::size_t i = 0; i < width; i++)
  {
    std::size_t j = 0;
    for (; j < radius; j++)
      out_p[i + j * width] = 0;

    for (; j < height - radius; j++)
    {
      out_p[i + j * width] = 0;
      for (int k = kw, l = static_cast<int> (j - radius); k >= 0; k--, l++)
        out_p[i + j * width] += in_p[i + l * width] * kptr[k];
    }

    for (; j < height; j++)
      out_p[i + j * width] = 0;
  }
}

#if defined(__RVV10__)

inline void
convolveRowsRVV (const float *in_p, float *out_p, std::size_t width, std::size_t height,
                 const float *kptr, std::size_t nkernel)
{
  const int kw = static_cast<int> (nkernel) - 1;
  const std::size_t radius = nkernel / 2;

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

inline void
convolveColsRVV (const float *in_p, float *out_p, std::size_t width, std::size_t height,
                 const float *kptr, std::size_t nkernel)
{
  const int kw = static_cast<int> (nkernel) - 1;
  const std::size_t radius = nkernel / 2;
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
      __builtin_prefetch (in_p + i + (j + vl) * width, 0, 3);

      vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2 (0.f, vl);
      for (int s = 0; s <= kw; ++s)
      {
        const float *const src = in_p + i + (j - radius + static_cast<std::size_t> (s)) * width;
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

} // namespace gaussian_convolve_local
