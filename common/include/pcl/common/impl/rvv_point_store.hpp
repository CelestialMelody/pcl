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

#ifdef __RVV10__
#include <type_traits>
#include <utility>
#endif

namespace pcl {
namespace rvv_store {

#ifdef __RVV10__

/** \note Deliberately duplicated from \c pcl::rvv_load (same semantics) so this header does not depend on
  * \c rvv_point_load.h; keep the two definitions in sync. */
template <typename T>
using RVVCoordScalar = std::remove_cv_t<std::remove_reference_t<T>>;

/** True if \a PointT is standard-layout and \c x, \c y, \c z are \c float (AoS \c offsetof + RVV \c f32 stores). */
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

template <std::size_t kStrideBytes>
inline void
strided_store_f32m2(float* field_ptr, const vfloat32m2_t v, const std::size_t vl)
{
  static_assert(kStrideBytes % alignof(float) == 0, "Stride must be aligned for float stores.");
  __riscv_vsse32_v_f32m2(field_ptr, static_cast<ptrdiff_t>(kStrideBytes), v, vl);
}

template <std::size_t kStrideBytes>
inline void
strided_store3_seg_f32m2(float* seg_base,
                         const std::size_t vl,
                         const vfloat32m2_t vx,
                         const vfloat32m2_t vy,
                         const vfloat32m2_t vz)
{
  static_assert(kStrideBytes % alignof(float) == 0, "Stride must be aligned for float stores.");
  vfloat32m2x3_t vt = __riscv_vset_v_f32m2_f32m2x3(__riscv_vundefined_f32m2x3(), 0, vx);
  vt = __riscv_vset_v_f32m2_f32m2x3(vt, 1, vy);
  vt = __riscv_vset_v_f32m2_f32m2x3(vt, 2, vz);
  __riscv_vssseg3e32_v_f32m2x3(seg_base, static_cast<ptrdiff_t>(kStrideBytes), vt, vl);
}

template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
strided_store3_fields_f32m2(std::uint8_t* base_u8,
                            const std::size_t vl,
                            const vfloat32m2_t vx,
                            const vfloat32m2_t vy,
                            const vfloat32m2_t vz)
{
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");
  float* px = reinterpret_cast<float*>(base_u8 + kXOff);
  float* py = reinterpret_cast<float*>(base_u8 + kYOff);
  float* pz = reinterpret_cast<float*>(base_u8 + kZOff);
  strided_store_f32m2<kStrideBytes>(px, vx, vl);
  strided_store_f32m2<kStrideBytes>(py, vy, vl);
  strided_store_f32m2<kStrideBytes>(pz, vz, vl);
}

template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
strided_store3_f32m2(const std::uint8_t* base_u8,
                     const std::size_t vl,
                     const vfloat32m2_t vx,
                     const vfloat32m2_t vy,
                     const vfloat32m2_t vz)
{
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");

  constexpr bool kTightlyPacked =
      (kYOff == kXOff + sizeof(float)) && (kZOff == kYOff + sizeof(float));

  std::uint8_t* mutable_base = const_cast<std::uint8_t*>(base_u8);
  if constexpr (kTightlyPacked) {
    float* seg_base = reinterpret_cast<float*>(mutable_base + kXOff);
    strided_store3_seg_f32m2<kStrideBytes>(seg_base, vl, vx, vy, vz);
  } else {
    strided_store3_fields_f32m2<kStrideBytes, kXOff, kYOff, kZOff>(mutable_base, vl, vx, vy, vz);
  }
}

template <std::size_t kStrideBytes>
inline void
strided_store4_seg_f32m2(float* seg_base,
                         const std::size_t vl,
                         const vfloat32m2_t v0,
                         const vfloat32m2_t v1,
                         const vfloat32m2_t v2,
                         const vfloat32m2_t v3)
{
  static_assert(kStrideBytes % alignof(float) == 0, "Stride must be aligned for float stores.");
  vfloat32m2x4_t vt = __riscv_vset_v_f32m2_f32m2x4(__riscv_vundefined_f32m2x4(), 0, v0);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 1, v1);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 2, v2);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 3, v3);
  __riscv_vssseg4e32_v_f32m2x4(seg_base, static_cast<ptrdiff_t>(kStrideBytes), vt, vl);
}

template <std::size_t kStrideBytes,
          std::size_t kF0Off,
          std::size_t kF1Off,
          std::size_t kF2Off,
          std::size_t kF3Off>
inline void
strided_store4_fields_f32m2(std::uint8_t* base_u8,
                            const std::size_t vl,
                            const vfloat32m2_t v0,
                            const vfloat32m2_t v1,
                            const vfloat32m2_t v2,
                            const vfloat32m2_t v3)
{
  static_assert(kF0Off % alignof(float) == 0 && kF1Off % alignof(float) == 0 &&
                    kF2Off % alignof(float) == 0 && kF3Off % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");
  float* p0 = reinterpret_cast<float*>(base_u8 + kF0Off);
  float* p1 = reinterpret_cast<float*>(base_u8 + kF1Off);
  float* p2 = reinterpret_cast<float*>(base_u8 + kF2Off);
  float* p3 = reinterpret_cast<float*>(base_u8 + kF3Off);
  strided_store_f32m2<kStrideBytes>(p0, v0, vl);
  strided_store_f32m2<kStrideBytes>(p1, v1, vl);
  strided_store_f32m2<kStrideBytes>(p2, v2, vl);
  strided_store_f32m2<kStrideBytes>(p3, v3, vl);
}

template <std::size_t kStrideBytes, std::size_t kF0Off, std::size_t kF1Off, std::size_t kF2Off, std::size_t kF3Off>
inline void
strided_store4_f32m2(const std::uint8_t* base_u8,
                     const std::size_t vl,
                     const vfloat32m2_t v0,
                     const vfloat32m2_t v1,
                     const vfloat32m2_t v2,
                     const vfloat32m2_t v3)
{
  static_assert(kF0Off % alignof(float) == 0 && kF1Off % alignof(float) == 0 &&
                    kF2Off % alignof(float) == 0 && kF3Off % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");

  constexpr bool kTightlyPacked = (kF1Off == kF0Off + sizeof(float)) &&
                                  (kF2Off == kF1Off + sizeof(float)) &&
                                  (kF3Off == kF2Off + sizeof(float));

  std::uint8_t* mutable_base = const_cast<std::uint8_t*>(base_u8);
  if constexpr (kTightlyPacked) {
    float* seg_base = reinterpret_cast<float*>(mutable_base + kF0Off);
    strided_store4_seg_f32m2<kStrideBytes>(seg_base, vl, v0, v1, v2, v3);
  } else {
    strided_store4_fields_f32m2<kStrideBytes, kF0Off, kF1Off, kF2Off, kF3Off>(
        mutable_base, vl, v0, v1, v2, v3);
  }
}

inline void
contiguous_store3_f32m2(float* dst0,
                        float* dst1,
                        float* dst2,
                        const std::size_t vl,
                        const vfloat32m2_t v0,
                        const vfloat32m2_t v1,
                        const vfloat32m2_t v2)
{
  __riscv_vse32_v_f32m2(dst0, v0, vl);
  __riscv_vse32_v_f32m2(dst1, v1, vl);
  __riscv_vse32_v_f32m2(dst2, v2, vl);
}

inline void
contiguous_seg3_store_f32m2(float* base,
                            const std::size_t vl,
                            const vfloat32m2_t v0,
                            const vfloat32m2_t v1,
                            const vfloat32m2_t v2)
{
  vfloat32m2x3_t vt = __riscv_vset_v_f32m2_f32m2x3(__riscv_vundefined_f32m2x3(), 0, v0);
  vt = __riscv_vset_v_f32m2_f32m2x3(vt, 1, v1);
  vt = __riscv_vset_v_f32m2_f32m2x3(vt, 2, v2);
  __riscv_vsseg3e32_v_f32m2x3(base, vt, vl);
}

inline void
contiguous_store4_f32m2(float* dst0,
                        float* dst1,
                        float* dst2,
                        float* dst3,
                        const std::size_t vl,
                        const vfloat32m2_t v0,
                        const vfloat32m2_t v1,
                        const vfloat32m2_t v2,
                        const vfloat32m2_t v3)
{
  __riscv_vse32_v_f32m2(dst0, v0, vl);
  __riscv_vse32_v_f32m2(dst1, v1, vl);
  __riscv_vse32_v_f32m2(dst2, v2, vl);
  __riscv_vse32_v_f32m2(dst3, v3, vl);
}

inline void
contiguous_seg4_store_f32m2(float* base,
                            const std::size_t vl,
                            const vfloat32m2_t v0,
                            const vfloat32m2_t v1,
                            const vfloat32m2_t v2,
                            const vfloat32m2_t v3)
{
  vfloat32m2x4_t vt = __riscv_vset_v_f32m2_f32m2x4(__riscv_vundefined_f32m2x4(), 0, v0);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 1, v1);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 2, v2);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 3, v3);
  __riscv_vsseg4e32_v_f32m2x4(base, vt, vl);
}

template <std::size_t kFieldOffBytes>
inline void
scatter_store_f32m2(std::uint8_t* base_u8,
              vuint32m2_t v_off_bytes,
              const vfloat32m2_t v,
              const std::size_t vl)
{
  static_assert(kFieldOffBytes % alignof(float) == 0, "Field offset must be aligned for float stores.");
  float* base_f32 = reinterpret_cast<float*>(base_u8 + kFieldOffBytes);
  __riscv_vsuxei32_v_f32m2(base_f32, v_off_bytes, v, vl);
}

inline void
scatter_store3_seg_f32m2(float* indexed_seg_base,
                 vuint32m2_t v_off_bytes,
                 const std::size_t vl,
                 const vfloat32m2_t vx,
                 const vfloat32m2_t vy,
                 const vfloat32m2_t vz)
{
  vfloat32m2x3_t vt = __riscv_vset_v_f32m2_f32m2x3(__riscv_vundefined_f32m2x3(), 0, vx);
  vt = __riscv_vset_v_f32m2_f32m2x3(vt, 1, vy);
  vt = __riscv_vset_v_f32m2_f32m2x3(vt, 2, vz);
  __riscv_vsuxseg3ei32_v_f32m2x3(indexed_seg_base, v_off_bytes, vt, vl);
}

template <std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
scatter_store3_fields_f32m2(std::uint8_t* base_u8,
                      vuint32m2_t v_off_bytes,
                      const std::size_t vl,
                      const vfloat32m2_t vx,
                      const vfloat32m2_t vy,
                      const vfloat32m2_t vz)
{
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");
  scatter_store_f32m2<kXOff>(base_u8, v_off_bytes, vx, vl);
  scatter_store_f32m2<kYOff>(base_u8, v_off_bytes, vy, vl);
  scatter_store_f32m2<kZOff>(base_u8, v_off_bytes, vz, vl);
}

template <std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
inline void
scatter_store3_f32m2(std::uint8_t* base_u8,
               vuint32m2_t v_off_bytes,
               const std::size_t vl,
               const vfloat32m2_t vx,
               const vfloat32m2_t vy,
               const vfloat32m2_t vz)
{
  static_assert(kXOff % alignof(float) == 0 && kYOff % alignof(float) == 0 && kZOff % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");

  constexpr bool kTightlyPacked =
      (kYOff == kXOff + sizeof(float)) && (kZOff == kYOff + sizeof(float));

  if constexpr (kTightlyPacked) {
    float* seg_base = reinterpret_cast<float*>(base_u8 + kXOff);
    scatter_store3_seg_f32m2(seg_base, v_off_bytes, vl, vx, vy, vz);
  } else {
    scatter_store3_fields_f32m2<kXOff, kYOff, kZOff>(base_u8, v_off_bytes, vl, vx, vy, vz);
  }
}

inline void
scatter_store4_seg_f32m2(float* indexed_seg_base,
                   vuint32m2_t v_off_bytes,
                   const std::size_t vl,
                   const vfloat32m2_t v0,
                   const vfloat32m2_t v1,
                   const vfloat32m2_t v2,
                   const vfloat32m2_t v3)
{
  vfloat32m2x4_t vt = __riscv_vset_v_f32m2_f32m2x4(__riscv_vundefined_f32m2x4(), 0, v0);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 1, v1);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 2, v2);
  vt = __riscv_vset_v_f32m2_f32m2x4(vt, 3, v3);
  __riscv_vsuxseg4ei32_v_f32m2x4(indexed_seg_base, v_off_bytes, vt, vl);
}

template <std::size_t kF0Off, std::size_t kF1Off, std::size_t kF2Off, std::size_t kF3Off>
inline void
scatter_store4_fields_f32m2(std::uint8_t* base_u8,
                      vuint32m2_t v_off_bytes,
                      const std::size_t vl,
                      const vfloat32m2_t v0,
                      const vfloat32m2_t v1,
                      const vfloat32m2_t v2,
                      const vfloat32m2_t v3)
{
  static_assert(kF0Off % alignof(float) == 0 && kF1Off % alignof(float) == 0 &&
                    kF2Off % alignof(float) == 0 && kF3Off % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");
  scatter_store_f32m2<kF0Off>(base_u8, v_off_bytes, v0, vl);
  scatter_store_f32m2<kF1Off>(base_u8, v_off_bytes, v1, vl);
  scatter_store_f32m2<kF2Off>(base_u8, v_off_bytes, v2, vl);
  scatter_store_f32m2<kF3Off>(base_u8, v_off_bytes, v3, vl);
}

template <std::size_t kF0Off, std::size_t kF1Off, std::size_t kF2Off, std::size_t kF3Off>
inline void
scatter_store4_f32m2(std::uint8_t* base_u8,
               vuint32m2_t v_off_bytes,
               const std::size_t vl,
               const vfloat32m2_t v0,
               const vfloat32m2_t v1,
               const vfloat32m2_t v2,
               const vfloat32m2_t v3)
{
  static_assert(kF0Off % alignof(float) == 0 && kF1Off % alignof(float) == 0 &&
                    kF2Off % alignof(float) == 0 && kF3Off % alignof(float) == 0,
                "Field offsets must be aligned for float stores.");

  constexpr bool kTightlyPacked = (kF1Off == kF0Off + sizeof(float)) &&
                                  (kF2Off == kF1Off + sizeof(float)) &&
                                  (kF3Off == kF2Off + sizeof(float));

  if constexpr (kTightlyPacked) {
    float* seg_base = reinterpret_cast<float*>(base_u8 + kF0Off);
    scatter_store4_seg_f32m2(seg_base, v_off_bytes, vl, v0, v1, v2, v3);
  } else {
    scatter_store4_fields_f32m2<kF0Off, kF1Off, kF2Off, kF3Off>(base_u8, v_off_bytes, vl, v0, v1, v2, v3);
  }
}

#endif // __RVV10__

} // namespace rvv_store
} // namespace pcl
