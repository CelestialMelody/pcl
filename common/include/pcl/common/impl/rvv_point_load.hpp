/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *
 *  Copyright (c) 2026
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
 */

#pragma once

#include <type_traits>
#include <utility>

namespace pcl {
namespace rvv_load {

#ifdef __RVV10__

/** Scalar behind \a PointT::x / \a y / \a z access (member or accessor; after cv/ref strip). */
template <typename T>
using RVVCoordScalar = std::remove_cv_t<std::remove_reference_t<T>>;

/** True if \a PointT is standard-layout and \c x, \c y, \c z are \c float — matches the assumptions of
  * \c offsetof(PointT, x|y|z) plus RVV \c f32 load/store helpers and their \c static_assert guards. */
template <typename PointT>
inline constexpr bool kRVVXYZPointCompatible =
    std::is_standard_layout_v<PointT> &&
    std::is_same_v<RVVCoordScalar<decltype (std::declval<PointT> ().x)>, float> &&
    std::is_same_v<RVVCoordScalar<decltype (std::declval<PointT> ().y)>, float> &&
    std::is_same_v<RVVCoordScalar<decltype (std::declval<PointT> ().z)>, float>;

template <typename T>
inline vuint32m2_t
byte_offsets_u32m2(vuint32m2_t v_idx, const std::size_t vl)
{
  static_assert(std::is_standard_layout_v<T>,
                "T must be standard-layout when used with offsetof()-style byte offsets.");
  return __riscv_vmul_vx_u32m2(v_idx, sizeof(T), vl);
}

template <typename T, std::size_t kFieldOffsetBytes>
inline vfloat32m2_t
gather_load_f32m2(const std::uint8_t* base_u8, vuint32m2_t v_off_bytes, const std::size_t vl)
{
  static_assert(std::is_standard_layout_v<T>,
                "T must be standard-layout if kFieldOffsetBytes is derived from offsetof(T, field).");
  static_assert(kFieldOffsetBytes % alignof(float) == 0, "Field offset must be aligned for float loads.");
  const float* base_f32 = reinterpret_cast<const float*>(base_u8 + kFieldOffsetBytes);
  return __riscv_vluxei32_v_f32m2(base_f32, v_off_bytes, vl);
}

template <std::size_t kStrideBytes>
inline vfloat32m2_t
strided_load_f32m2(const float* field_ptr, const std::size_t vl)
{
  static_assert(kStrideBytes % alignof(float) == 0, "Stride must be aligned for float loads.");
  return __riscv_vlse32_v_f32m2(field_ptr, static_cast<ptrdiff_t>(kStrideBytes), vl);
}

template <std::size_t kStrideBytes>
inline void
strided_load3_seg_f32m2(const float* seg_base,
                        const std::size_t vl,
                        vfloat32m2_t& vx,
                        vfloat32m2_t& vy,
                        vfloat32m2_t& vz)
{
  static_assert(kStrideBytes % alignof(float) == 0, "Stride must be aligned for float loads.");
  const vfloat32m2x3_t v_xyz =
      __riscv_vlsseg3e32_v_f32m2x3(seg_base, static_cast<ptrdiff_t>(kStrideBytes), vl);
  vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
  vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
  vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);
}

template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
strided_load3_fields_f32m2(const std::uint8_t* base_u8,
                           const std::size_t vl,
                           vfloat32m2_t& vx,
                           vfloat32m2_t& vy,
                           vfloat32m2_t& vz)
{
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float loads.");
  const float* xptr = reinterpret_cast<const float*>(base_u8 + kXOff);
  const float* yptr = reinterpret_cast<const float*>(base_u8 + kYOff);
  const float* zptr = reinterpret_cast<const float*>(base_u8 + kZOff);
  vx = strided_load_f32m2<kStrideBytes>(xptr, vl);
  vy = strided_load_f32m2<kStrideBytes>(yptr, vl);
  vz = strided_load_f32m2<kStrideBytes>(zptr, vl);
}

inline void
indexed_load3_seg_f32m2(const float* indexed_seg_base,
                        vuint32m2_t v_off_bytes,
                        const std::size_t vl,
                        vfloat32m2_t& vx,
                        vfloat32m2_t& vy,
                        vfloat32m2_t& vz)
{
  const vfloat32m2x3_t v_xyz = __riscv_vluxseg3ei32_v_f32m2x3(indexed_seg_base, v_off_bytes, vl);
  vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
  vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
  vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);
}

template <typename T, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
indexed_load3_fields_f32m2(const std::uint8_t* base_u8,
                           vuint32m2_t v_off_bytes,
                           const std::size_t vl,
                           vfloat32m2_t& vx,
                           vfloat32m2_t& vy,
                           vfloat32m2_t& vz)
{
  static_assert(std::is_standard_layout_v<T>,
                "T must be standard-layout if offsets are derived from offsetof(T, field).");
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float loads.");
  vx = gather_load_f32m2<T, kXOff>(base_u8, v_off_bytes, vl);
  vy = gather_load_f32m2<T, kYOff>(base_u8, v_off_bytes, vl);
  vz = gather_load_f32m2<T, kZOff>(base_u8, v_off_bytes, vl);
}

inline void
contiguous_seg3_load_f32m2(const float* base_f32,
                           const std::size_t vl,
                           vfloat32m2_t& vx,
                           vfloat32m2_t& vy,
                           vfloat32m2_t& vz)
{
  const vfloat32m2x3_t v_xyz = __riscv_vlseg3e32_v_f32m2x3(base_f32, vl);
  vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
  vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
  vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);
}

template <typename T, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
indexed_load3_f32m2(const std::uint8_t* base_u8,
                    vuint32m2_t v_off_bytes,
                    const std::size_t vl,
                    vfloat32m2_t& vx,
                    vfloat32m2_t& vy,
                    vfloat32m2_t& vz)
{
  static_assert(std::is_standard_layout_v<T>,
                "T must be standard-layout if offsets are derived from offsetof(T, field).");
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float loads.");

  constexpr bool kTightlyPacked =
      (kYOff == kXOff + sizeof(float)) && (kZOff == kYOff + sizeof(float));

  if constexpr (kTightlyPacked) {
    const float* base_f32 = reinterpret_cast<const float*>(base_u8 + kXOff);
    indexed_load3_seg_f32m2(base_f32, v_off_bytes, vl, vx, vy, vz);
  } else {
    indexed_load3_fields_f32m2<T, kXOff, kYOff, kZOff>(base_u8, v_off_bytes, vl, vx, vy, vz);
  }
}

template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
strided_load3_f32m2(const std::uint8_t* base_u8,
                    const std::size_t vl,
                    vfloat32m2_t& vx,
                    vfloat32m2_t& vy,
                    vfloat32m2_t& vz)
{
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float loads.");

  constexpr bool kTightlyPacked =
      (kYOff == kXOff + sizeof(float)) && (kZOff == kYOff + sizeof(float));

  if constexpr (kTightlyPacked) {
    const float* seg_base = reinterpret_cast<const float*>(base_u8 + kXOff);
    strided_load3_seg_f32m2<kStrideBytes>(seg_base, vl, vx, vy, vz);
  } else {
    strided_load3_fields_f32m2<kStrideBytes, kXOff, kYOff, kZOff>(base_u8, vl, vx, vy, vz);
  }
}

#endif // __RVV10__

} // namespace rvv_load
} // namespace pcl
