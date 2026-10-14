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

/** \brief RVV point/field store helpers.
 *
 * API layers (symmetric with `pcl::rvv_load` / `rvv_point_load.h`):
 * - **Primitives (`*_seg_*`, `*_fields_*`, `strided_store_f32m2`,
 * `masked_strided_store_f32m2`, `scatter_store_f32m2`)** each map to one fixed
 * instruction strategy (no `if constexpr` dispatch). Use in micro-benchmarks to compare
 *   e.g. `vssseg3e32` vs 3×`vsse32`, or `vsuxseg3ei32` vs 3×`vsuxei32`, fairly.
 * - **Dispatch** (`strided_store3_f32m2`, `strided_store4_f32m2`,
 * `scatter_store3_f32m2`, `scatter_store4_f32m2`) choose seg vs fields at compile time
 * when the float fields are tightly packed in the AoS struct.
 * - **SoA / packed contiguous**: `contiguous_store{3,4}_f32m2`,
 * `contiguous_seg{3,4}_store_f32m2`.
 *
 * Intrinsic symmetry with loads: `vlse32`↔`vsse32`, `vlsseg*`↔`vssseg*`,
 * `vlseg*`↔`vsseg*`, `vlux*` / `vluxseg*` ↔ `vsux*` / `vsuxseg*`. Point traits (\c
 * RVVCoordScalar, \c kRVVXYZPointCompatible) are thin aliases to
 * \c pcl/rvv_point_traits.h with the same semantics as in \c rvv_point_load.
 *
 * A single-field indexed store is named `scatter_store_f32m2`, the counterpart of
 * `gather_load_f32m2`.
 *
 * \ingroup common
 */
namespace rvv_store {

#ifdef __RVV10__

template <typename T>
vuint32m2_t
byte_offsets_u32m2(vuint32m2_t v_idx, std::size_t vl);

template <std::size_t kStrideBytes>
void
strided_store_f32m2(float* field_ptr, vfloat32m2_t v, std::size_t vl);

/** \brief Masked single-field strided store (`vsse32_m`). */
template <std::size_t kStrideBytes>
void
masked_strided_store_f32m2(vbool16_t mask,
                           float* field_ptr,
                           vfloat32m2_t v,
                           std::size_t vl);

/** \brief PCL field-tag strided store for one registered single-float field. */
template <typename PointT, typename Field>
void
strided_store_field_f32m2(std::uint8_t* base_u8, vfloat32m2_t v, std::size_t vl);

/** \brief `vssseg3e32`; `seg_base` = first of three consecutive destination `float`s.
 */
template <std::size_t kStrideBytes>
void
strided_store3_seg_f32m2(
    float* seg_base, std::size_t vl, vfloat32m2_t vx, vfloat32m2_t vy, vfloat32m2_t vz);

/** \brief 3×`vsse32` at `kXOff`/`kYOff`/`kZOff` (no segment store). */
template <std::size_t kStrideBytes,
          std::size_t kXOff,
          std::size_t kYOff,
          std::size_t kZOff>
void
strided_store3_fields_f32m2(std::uint8_t* base_u8,
                            std::size_t vl,
                            vfloat32m2_t vx,
                            vfloat32m2_t vy,
                            vfloat32m2_t vz);

template <std::size_t kStrideBytes,
          std::size_t kXOff,
          std::size_t kYOff,
          std::size_t kZOff>
void
strided_store3_f32m2(const std::uint8_t* base_u8,
                     std::size_t vl,
                     vfloat32m2_t vx,
                     vfloat32m2_t vy,
                     vfloat32m2_t vz);

/** \brief `vssseg4e32`; `seg_base` = first of four consecutive destination `float`s. */
template <std::size_t kStrideBytes>
void
strided_store4_seg_f32m2(float* seg_base,
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
void
strided_store4_fields_f32m2(std::uint8_t* base_u8,
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
void
strided_store4_f32m2(const std::uint8_t* base_u8,
                     std::size_t vl,
                     vfloat32m2_t v0,
                     vfloat32m2_t v1,
                     vfloat32m2_t v2,
                     vfloat32m2_t v3);

void
contiguous_store3_f32m2(float* dst0,
                        float* dst1,
                        float* dst2,
                        std::size_t vl,
                        vfloat32m2_t v0,
                        vfloat32m2_t v1,
                        vfloat32m2_t v2);

void
contiguous_seg3_store_f32m2(
    float* base, std::size_t vl, vfloat32m2_t v0, vfloat32m2_t v1, vfloat32m2_t v2);

void
contiguous_store4_f32m2(float* dst0,
                        float* dst1,
                        float* dst2,
                        float* dst3,
                        std::size_t vl,
                        vfloat32m2_t v0,
                        vfloat32m2_t v1,
                        vfloat32m2_t v2,
                        vfloat32m2_t v3);

void
contiguous_seg4_store_f32m2(float* base,
                            std::size_t vl,
                            vfloat32m2_t v0,
                            vfloat32m2_t v1,
                            vfloat32m2_t v2,
                            vfloat32m2_t v3);

template <std::size_t kFieldOffBytes>
void
scatter_store_f32m2(std::uint8_t* base_u8,
                    vuint32m2_t v_off_bytes,
                    vfloat32m2_t v,
                    std::size_t vl);

/** \brief PCL field-tag indexed scatter for one registered single-float field. */
template <typename PointT, typename Field>
void
scatter_store_field_f32m2(std::uint8_t* base_u8,
                          vuint32m2_t v_off_bytes,
                          vfloat32m2_t v,
                          std::size_t vl);

void
scatter_store3_seg_f32m2(float* indexed_seg_base,
                         vuint32m2_t v_off_bytes,
                         std::size_t vl,
                         vfloat32m2_t vx,
                         vfloat32m2_t vy,
                         vfloat32m2_t vz);

template <std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void
scatter_store3_fields_f32m2(std::uint8_t* base_u8,
                            vuint32m2_t v_off_bytes,
                            std::size_t vl,
                            vfloat32m2_t vx,
                            vfloat32m2_t vy,
                            vfloat32m2_t vz);

template <std::size_t kXOff, std::size_t kYOff, std::size_t kZOff>
void
scatter_store3_f32m2(std::uint8_t* base_u8,
                     vuint32m2_t v_off_bytes,
                     std::size_t vl,
                     vfloat32m2_t vx,
                     vfloat32m2_t vy,
                     vfloat32m2_t vz);

void
scatter_store4_seg_f32m2(float* indexed_seg_base,
                         vuint32m2_t v_off_bytes,
                         std::size_t vl,
                         vfloat32m2_t v0,
                         vfloat32m2_t v1,
                         vfloat32m2_t v2,
                         vfloat32m2_t v3);

template <std::size_t kF0Off,
          std::size_t kF1Off,
          std::size_t kF2Off,
          std::size_t kF3Off>
void
scatter_store4_fields_f32m2(std::uint8_t* base_u8,
                            vuint32m2_t v_off_bytes,
                            std::size_t vl,
                            vfloat32m2_t v0,
                            vfloat32m2_t v1,
                            vfloat32m2_t v2,
                            vfloat32m2_t v3);

template <std::size_t kF0Off,
          std::size_t kF1Off,
          std::size_t kF2Off,
          std::size_t kF3Off>
void
scatter_store4_f32m2(std::uint8_t* base_u8,
                     vuint32m2_t v_off_bytes,
                     std::size_t vl,
                     vfloat32m2_t v0,
                     vfloat32m2_t v1,
                     vfloat32m2_t v2,
                     vfloat32m2_t v3);

#endif // __RVV10__

} // namespace rvv_store
} // namespace pcl

#include <pcl/impl/rvv_point_store.hpp>
