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

/** \brief RVV point/field load helpers.
  *
  * API layers:
  * - **Primitives (`*_seg_*`, `*_fields_*`, `strided_load_f32m2`, `gather_load_f32m2`)** map to one
  *   fixed instruction strategy each (no `if constexpr` dispatch). Use these in micro-benchmarks to
  *   compare e.g. `vlsseg3e32` vs 3×`vlse32`, or `vluxseg3ei32` vs 3×`vluxei32`, fairly.
  * - **Dispatch (`strided_load3_f32m2`, `indexed_load3_f32m2`)** choose seg vs fields at compile time
  *   when x/y/z are consecutive in memory.
  * - **Point traits** (`RVVCoordScalar`, `kRVVXYZPointCompatible`) in the implementation header gate
  *   `offsetof(PointT, x|y|z)` + float RVV paths; they align with the \c static_assert checks on stride/alignment.
  * - **3× indexed gather** (no segment instruction) is **`indexed_load3_fields_f32m2`**.
  * - **`gather_load_f32m2`** implements single-field `vluxei32` (indexed gather).
  *
  * 中文：分为「原子原语」（固定一种指令，便于 bench 公平对比）与「带编译期分发的合并接口」。
  * `*_seg_*` 表示 segment 指令路径；`*_fields_*` 表示按字段分别加载（3× strided 或 3× gather）。
  *
  * \ingroup common
  */
namespace rvv_load {

#ifdef __RVV10__

template <typename T>
vuint32m2_t byte_offsets_u32m2(vuint32m2_t v_idx, std::size_t vl);

/** \brief Single-field indexed gather (`vluxei32`). \ingroup common */
template <typename T, std::size_t kFieldOffsetBytes>
vfloat32m2_t gather_load_f32m2(const std::uint8_t* base_u8, vuint32m2_t v_off_bytes, std::size_t vl);

template <std::size_t kStrideBytes>
vfloat32m2_t strided_load_f32m2(const float* field_ptr, std::size_t vl);

/** \brief `vlsseg3e32`: `seg_base` must point to the first of three consecutive `float`s. */
template <std::size_t kStrideBytes>
void strided_load3_seg_f32m2(const float* seg_base,
                             std::size_t vl,
                             vfloat32m2_t& vx,
                             vfloat32m2_t& vy,
                             vfloat32m2_t& vz);

/** \brief 3×`vlse32` at `kXOff`/`kYOff`/`kZOff` (no segment instruction). */
template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void strided_load3_fields_f32m2(const std::uint8_t* base_u8,
                                std::size_t vl,
                                vfloat32m2_t& vx,
                                vfloat32m2_t& vy,
                                vfloat32m2_t& vz);

/** \brief `vluxseg3ei32`: `indexed_seg_base` is the indexed base for the first of three consecutive floats. */
void indexed_load3_seg_f32m2(const float* indexed_seg_base,
                             vuint32m2_t v_off_bytes,
                             std::size_t vl,
                             vfloat32m2_t& vx,
                             vfloat32m2_t& vy,
                             vfloat32m2_t& vz);

/** \brief 3×`vluxei32` at the given field offsets (no segment instruction). */
template <typename T, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void indexed_load3_fields_f32m2(const std::uint8_t* base_u8,
                                vuint32m2_t v_off_bytes,
                                std::size_t vl,
                                vfloat32m2_t& vx,
                                vfloat32m2_t& vy,
                                vfloat32m2_t& vz);

void contiguous_seg3_load_f32m2(const float* base_f32,
                                std::size_t vl,
                                vfloat32m2_t& vx,
                                vfloat32m2_t& vy,
                                vfloat32m2_t& vz);

template <typename T, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void indexed_load3_f32m2(const std::uint8_t* base_u8,
                         vuint32m2_t v_off_bytes,
                         std::size_t vl,
                         vfloat32m2_t& vx,
                         vfloat32m2_t& vy,
                         vfloat32m2_t& vz);

template <std::size_t kStrideBytes, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void strided_load3_f32m2(const std::uint8_t* base_u8,
                         std::size_t vl,
                         vfloat32m2_t& vx,
                         vfloat32m2_t& vy,
                         vfloat32m2_t& vz);

#endif // __RVV10__

} // namespace rvv_load
} // namespace pcl

#include <pcl/common/impl/rvv_point_load.hpp>

