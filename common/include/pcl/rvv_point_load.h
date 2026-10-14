/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2026-, Open Perception Inc.
 *
 *  All rights reserved
 */

#pragma once

#include <pcl/rvv_point_traits.h>

#include <cstddef>
#include <cstdint>

#ifdef __RVV10__
#include <riscv_vector.h>
#endif

namespace pcl {

/** \brief RVV point/field load helpers.
 *
 * API layers:
 * - **Primitives (`*_seg_*`, `*_fields_*`, `strided_load_f32m2`, `gather_load_f32m2`)**
 * map to one fixed instruction strategy each (no `if constexpr` dispatch). Use these in
 * micro-benchmarks to compare e.g. `vlsseg3e32` vs 3×`vlse32`, or `vluxseg3ei32` vs
 * 3×`vluxei32`, fairly.
 * - **Dispatch** keeps the access-pattern-specific defaults: `strided_load3_f32m2`
 *   selects seg for consecutive fields, while `indexed_load3_f32m2` uses three
 *   field-wise gathers. Explicit `*_seg_*` and `*_fields_*` primitives remain available.
 * - **Point traits** (`RVVCoordScalar`, `kRVVXYZPointCompatible`) are thin aliases to
 *   `pcl/rvv_point_traits.h`; they gate
 *   `offsetof(PointT, x|y|z)` + float RVV paths; they align with the \c static_assert
 * checks on stride/alignment.
 * - **3× indexed gather** (no segment instruction) is **`indexed_load3_fields_f32m2`**.
 * - **`gather_load_f32m2`** implements single-field `vluxei32` (indexed gather).
 *
 * The `*_seg_*` names take the segment instruction path; the `*_fields_*` names load
 * the fields separately (three strided or three gathered accesses), which is what makes
 * the two strategies comparable.
 *
 * \ingroup common
 */
namespace rvv_load {

#ifdef __RVV10__

template <typename T>
vuint32m2_t
byte_offsets_u32m2(vuint32m2_t v_idx, std::size_t vl);

/** \brief Single-field indexed gather (`vluxei32`). \ingroup common */
template <typename T, std::size_t kFieldOffsetBytes>
vfloat32m2_t
gather_load_f32m2(const std::uint8_t* base_u8, vuint32m2_t v_off_bytes, std::size_t vl);

template <std::size_t kStrideBytes>
vfloat32m2_t
strided_load_f32m2(const float* field_ptr, std::size_t vl);

/** \brief PCL field-tag strided load for one registered single-float field. */
template <typename PointT, typename Field>
vfloat32m2_t
strided_load_field_f32m2(const std::uint8_t* base_u8, std::size_t vl);

/** \brief PCL field-tag indexed gather for one registered single-float field. */
template <typename PointT, typename Field>
vfloat32m2_t
indexed_load_field_f32m2(const std::uint8_t* base_u8,
                         vuint32m2_t v_off_bytes,
                         std::size_t vl);

/** \brief `vlsseg3e32`: `seg_base` must point to the first of three consecutive
 * `float`s. */
template <std::size_t kStrideBytes>
void
strided_load3_seg_f32m2(const float* seg_base,
                        std::size_t vl,
                        vfloat32m2_t& vx,
                        vfloat32m2_t& vy,
                        vfloat32m2_t& vz);

/** \brief 3×`vlse32` at `kXOff`/`kYOff`/`kZOff` (no segment instruction). */
template <std::size_t kStrideBytes,
          std::size_t kXOff,
          std::size_t kYOff,
          std::size_t kZOff>
void
strided_load3_fields_f32m2(const std::uint8_t* base_u8,
                           std::size_t vl,
                           vfloat32m2_t& vx,
                           vfloat32m2_t& vy,
                           vfloat32m2_t& vz);

/** \brief `vluxseg3ei32`: `indexed_seg_base` is the indexed base for the first of three
 * consecutive floats. */
void
indexed_load3_seg_f32m2(const float* indexed_seg_base,
                        vuint32m2_t v_off_bytes,
                        std::size_t vl,
                        vfloat32m2_t& vx,
                        vfloat32m2_t& vy,
                        vfloat32m2_t& vz);

/** \brief 3×`vluxei32` at the given field offsets (no segment instruction). */
template <typename T, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void
indexed_load3_fields_f32m2(const std::uint8_t* base_u8,
                           vuint32m2_t v_off_bytes,
                           std::size_t vl,
                           vfloat32m2_t& vx,
                           vfloat32m2_t& vy,
                           vfloat32m2_t& vz);

void
contiguous_seg3_load_f32m2(const float* base_f32,
                           std::size_t vl,
                           vfloat32m2_t& vx,
                           vfloat32m2_t& vy,
                           vfloat32m2_t& vz);

/** \brief Default three-field indexed load using three field-wise gathers. */
template <typename T, std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void
indexed_load3_f32m2(const std::uint8_t* base_u8,
                    vuint32m2_t v_off_bytes,
                    std::size_t vl,
                    vfloat32m2_t& vx,
                    vfloat32m2_t& vy,
                    vfloat32m2_t& vz);

template <std::size_t kStrideBytes,
          std::size_t kXOff,
          std::size_t kYOff,
          std::size_t kZOff>
void
strided_load3_f32m2(const std::uint8_t* base_u8,
                    std::size_t vl,
                    vfloat32m2_t& vx,
                    vfloat32m2_t& vy,
                    vfloat32m2_t& vz);

#endif // __RVV10__

} // namespace rvv_load
} // namespace pcl

#include <pcl/impl/rvv_point_load.hpp>
