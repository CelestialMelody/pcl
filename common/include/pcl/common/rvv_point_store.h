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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef __RVV10__
#include <riscv_vector.h>
#endif

namespace pcl {

/** \brief RVV point/field store helpers.
  *
  * API layers (symmetric with `pcl::rvv_load` / `rvv_point_load.h`):
  * - **Primitives (`*_seg_*`, `*_fields_*`, `strided_store_f32m2`, `scatter_store_f32m2`)** each map to one
  *   fixed instruction strategy (no `if constexpr` dispatch). Use in micro-benchmarks to compare
  *   e.g. `vssseg3e32` vs 3×`vsse32`, or `vsuxseg3ei32` vs 3×`vsuxei32`, fairly.
  * - **Dispatch** (`strided_store3_f32m2`, `strided_store4_f32m2`, `scatter_store3_f32m2`, `scatter_store4_f32m2`) choose seg vs fields at compile
  *   time when the float fields are tightly packed in the AoS struct.
  * - **SoA / packed contiguous**: `contiguous_store{3,4}_f32m2`, `contiguous_seg{3,4}_store_f32m2`.
  *
  * Intrinsic symmetry with loads: `vlse32`↔`vsse32`, `vlsseg*`↔`vssseg*`, `vlseg*`↔`vsseg*`,
  * `vlux*` / `vluxseg*` ↔ `vsux*` / `vsuxseg*`.
  *
  * 中文：与 `rvv_point_load` 对称——原语层固定一种 store 指令；合并接口在编译期判断字段是否紧密连续。
  * Indexed 单字段写回命名 `scatter_store_f32m2`，与 `gather_load_f32m2` 对仗。
  *
  * \ingroup common
  */
namespace rvv_store {

#ifdef __RVV10__

template <typename T>
vuint32m2_t byte_offsets_u32m2(vuint32m2_t v_idx, std::size_t vl);

template <std::size_t kStrideBytes>
void strided_store_f32m2(float* field_ptr, vfloat32m2_t v, std::size_t vl);

/** \brief `vssseg3e32`; `seg_base` = first of three consecutive destination `float`s. */
template <std::size_t kStrideBytes>
void strided_store3_seg_f32m2(float* seg_base,
                              std::size_t vl,
                              vfloat32m2_t vx,
                              vfloat32m2_t vy,
                              vfloat32m2_t vz);

/** \brief 3×`vsse32` at `kXOff`/`kYOff`/`kZOff` (no segment store). */
template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void strided_store3_fields_f32m2(std::uint8_t* base_u8,
                                 std::size_t vl,
                                 vfloat32m2_t vx,
                                 vfloat32m2_t vy,
                                 vfloat32m2_t vz);

template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void strided_store3_f32m2(const std::uint8_t* base_u8,
                          std::size_t vl,
                          vfloat32m2_t vx,
                          vfloat32m2_t vy,
                          vfloat32m2_t vz);

/** \brief `vssseg4e32`; `seg_base` = first of four consecutive destination `float`s. */
template <std::size_t kStrideBytes>
void strided_store4_seg_f32m2(float* seg_base,
                              std::size_t vl,
                              vfloat32m2_t v0,
                              vfloat32m2_t v1,
                              vfloat32m2_t v2,
                              vfloat32m2_t v3);

template <std::size_t kStrideBytes,
          std::size_t kF0Off,
          std::size_t kF1Off,
          std::size_t kF2Off,
          std::size_t kF3Off>
void strided_store4_fields_f32m2(std::uint8_t* base_u8,
                                 std::size_t vl,
                                 vfloat32m2_t v0,
                                 vfloat32m2_t v1,
                                 vfloat32m2_t v2,
                                 vfloat32m2_t v3);

template <std::size_t kStrideBytes, std::size_t kF0Off, std::size_t kF1Off, std::size_t kF2Off, std::size_t kF3Off>
void strided_store4_f32m2(const std::uint8_t* base_u8,
                          std::size_t vl,
                          vfloat32m2_t v0,
                          vfloat32m2_t v1,
                          vfloat32m2_t v2,
                          vfloat32m2_t v3);

void contiguous_store3_f32m2(float* dst0,
                             float* dst1,
                             float* dst2,
                             std::size_t vl,
                             vfloat32m2_t v0,
                             vfloat32m2_t v1,
                             vfloat32m2_t v2);

void contiguous_seg3_store_f32m2(float* base,
                                 std::size_t vl,
                                 vfloat32m2_t v0,
                                 vfloat32m2_t v1,
                                 vfloat32m2_t v2);

void contiguous_store4_f32m2(float* dst0,
                             float* dst1,
                             float* dst2,
                             float* dst3,
                             std::size_t vl,
                             vfloat32m2_t v0,
                             vfloat32m2_t v1,
                             vfloat32m2_t v2,
                             vfloat32m2_t v3);

void contiguous_seg4_store_f32m2(float* base,
                                 std::size_t vl,
                                 vfloat32m2_t v0,
                                 vfloat32m2_t v1,
                                 vfloat32m2_t v2,
                                 vfloat32m2_t v3);

template <std::size_t kFieldOffBytes>
void scatter_store_f32m2(std::uint8_t* base_u8, vuint32m2_t v_off_bytes, vfloat32m2_t v, std::size_t vl);

void scatter_store3_seg_f32m2(float* indexed_seg_base,
                        vuint32m2_t v_off_bytes,
                        std::size_t vl,
                        vfloat32m2_t vx,
                        vfloat32m2_t vy,
                        vfloat32m2_t vz);

template <std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void scatter_store3_fields_f32m2(std::uint8_t* base_u8,
                           vuint32m2_t v_off_bytes,
                           std::size_t vl,
                           vfloat32m2_t vx,
                           vfloat32m2_t vy,
                           vfloat32m2_t vz);

template <std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void scatter_store3_f32m2(std::uint8_t* base_u8,
                    vuint32m2_t v_off_bytes,
                    std::size_t vl,
                    vfloat32m2_t vx,
                    vfloat32m2_t vy,
                    vfloat32m2_t vz);

void scatter_store4_seg_f32m2(float* indexed_seg_base,
                        vuint32m2_t v_off_bytes,
                        std::size_t vl,
                        vfloat32m2_t v0,
                        vfloat32m2_t v1,
                        vfloat32m2_t v2,
                        vfloat32m2_t v3);

template <std::size_t kF0Off, std::size_t kF1Off, std::size_t kF2Off, std::size_t kF3Off>
void scatter_store4_fields_f32m2(std::uint8_t* base_u8,
                           vuint32m2_t v_off_bytes,
                           std::size_t vl,
                           vfloat32m2_t v0,
                           vfloat32m2_t v1,
                           vfloat32m2_t v2,
                           vfloat32m2_t v3);

template <std::size_t kF0Off, std::size_t kF1Off, std::size_t kF2Off, std::size_t kF3Off>
void scatter_store4_f32m2(std::uint8_t* base_u8,
                    vuint32m2_t v_off_bytes,
                    std::size_t vl,
                    vfloat32m2_t v0,
                    vfloat32m2_t v1,
                    vfloat32m2_t v2,
                    vfloat32m2_t v3);

#endif // __RVV10__

} // namespace rvv_store
} // namespace pcl

#include <pcl/common/impl/rvv_point_store.hpp>
