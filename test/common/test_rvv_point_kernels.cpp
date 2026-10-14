/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2026-, Open Perception Inc.
 *
 *  All rights reserved
 */

#include <pcl/test/gtest.h>
#include <pcl/point_types.h>
#include <pcl/rvv_point_traits.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__RVV10__)
#include <pcl/rvv_point_load.h>
#include <pcl/rvv_point_store.h>

#include <riscv_vector.h>
#endif

// The layout gates only read PCL field-registration metadata, so they answer the
// same question in a build without the vector extension. These assertions are the
// portable half of this topic's behavior.
static_assert(pcl::rvv::RVVXYZFloatLayout<pcl::PointXYZ>::value,
              "PointXYZ registers single-float x/y/z");
static_assert(pcl::rvv::kRVVXYZPointCompatible<pcl::PointXYZI>,
              "PointXYZI also registers single-float x/y/z");
static_assert(!pcl::rvv::RVVXYZFloatLayout<pcl::PointXY>::value,
              "PointXY has no z field");
static_assert(pcl::rvv::RVVXYZAoSFloatLayout<pcl::PointXYZ>::value,
              "PointXYZ is a standard-layout AoS point with aligned float fields");
static_assert(pcl::rvv::kRVVXYZNormalPointCompatible<pcl::PointNormal>,
              "PointNormal registers single-float x/y/z and normal_x/y/z");
static_assert(pcl::rvv::RVVXYZNormalFloatLayout<pcl::PointXYZINormal>::value,
              "PointXYZINormal has AoS single-float xyz and normal fields");
static_assert(pcl::rvv::RVVNormalFloatLayout<pcl::Normal>::value,
              "Normal registers single-float normal fields");
static_assert(
    pcl::rvv::RVVFloatFieldLayout<pcl::PointXYZI, pcl::fields::intensity>::value,
    "PointXYZI registers a single-float intensity field");
static_assert(
    pcl::rvv::RVVFloatFieldLayout<pcl::PointXYZINormal, pcl::fields::curvature>::value,
    "PointXYZINormal registers a single-float curvature field");
static_assert(
    !pcl::rvv::RVVFloatFieldLayout<pcl::PointXYZI, pcl::fields::normal_x>::value,
    "An unregistered field must not satisfy the field-tag gate");
static_assert(!pcl::rvv::RVVFloatFieldLayout<pcl::PointXY, pcl::fields::z>::value,
              "PointXY has no registered z field");
static_assert(!pcl::rvv::RVVFloatFieldLayout<pcl::Normal, pcl::fields::x>::value,
              "Normal has no registered x field");

TEST(RVVPointTraits, ReportsFieldOffsetsOfTheQueriedPointType)
{
  // The offsets describe PointT, so a caller that reads one point type and writes
  // another has to gate both of them separately.
  EXPECT_EQ(pcl::rvv::RVVXYZAoSFloatLayout<pcl::PointXYZ>::kX,
            offsetof(pcl::PointXYZ, x));
  EXPECT_EQ(pcl::rvv::RVVXYZAoSFloatLayout<pcl::PointXYZ>::kY,
            offsetof(pcl::PointXYZ, y));
  EXPECT_EQ(pcl::rvv::RVVXYZAoSFloatLayout<pcl::PointXYZ>::kZ,
            offsetof(pcl::PointXYZ, z));
}

TEST(RVVPointTraits, BoundsTheIndexedOffsetRepresentation)
{
  // Byte offsets are carried in 32-bit vectors, so this bound is what tells a
  // caller whether every legal index of a cloud is still representable.
  EXPECT_EQ(pcl::rvv::rvvMaxU32ByteOffsetElements<pcl::PointXYZ>(),
            std::numeric_limits<std::uint32_t>::max() / sizeof(pcl::PointXYZ));
  EXPECT_LT(pcl::rvv::rvvMaxU32ByteOffsetElements<pcl::PointNormal>(),
            pcl::rvv::rvvMaxU32ByteOffsetElements<pcl::PointXYZ>());
}

#if defined(__RVV10__)

namespace {

// A deterministic AoS cloud: each field differs from the others so a helper that
// reads the wrong offset cannot pass by coincidence.
std::vector<pcl::PointXYZI>
makePoints(const std::size_t count)
{
  std::vector<pcl::PointXYZI> points(count);
  for (std::size_t i = 0; i < count; ++i) {
    points[i].x = static_cast<float>(i);
    points[i].y = static_cast<float>(i) + 0.25f;
    points[i].z = static_cast<float>(i) + 0.5f;
    points[i].intensity = static_cast<float>(i) + 0.75f;
  }
  return points;
}

// PointXYZINormal deliberately keeps the xyz and normal fields separated by
// additional data. This makes the field-tag tests catch an offset that is
// accidentally assumed to be tightly packed.
std::vector<pcl::PointXYZINormal>
makeNormalPoints(const std::size_t count)
{
  std::vector<pcl::PointXYZINormal> points(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float value = static_cast<float>(i);
    points[i].x = value + 0.1f;
    points[i].y = value + 0.2f;
    points[i].z = value + 0.3f;
    points[i].intensity = value + 0.4f;
    points[i].normal_x = value + 1.1f;
    points[i].normal_y = value + 1.2f;
    points[i].normal_z = value + 1.3f;
    points[i].curvature = value + 1.4f;
  }
  return points;
}

std::vector<float>
readVector(const vfloat32m2_t v, const std::size_t vl)
{
  std::vector<float> values(vl);
  __riscv_vse32_v_f32m2(values.data(), v, vl);
  return values;
}

// Compile-only field-tag coverage mirrors the public helper contract for every
// representative registered point type. The function is never called: keeping
// these instantiations out of runtime tests ensures compile coverage cannot be
// mistaken for execution coverage.
template <typename PointT, typename Field>
[[maybe_unused]] void
instantiate_field_tag_api(std::uint8_t* base_u8,
                          const vuint32m2_t v_indices,
                          const std::size_t vl)
{
  const vuint32m2_t offsets = pcl::rvv_load::byte_offsets_u32m2<PointT>(v_indices, vl);
  const vfloat32m2_t strided =
      pcl::rvv_load::strided_load_field_f32m2<PointT, Field>(base_u8, vl);
  const vfloat32m2_t indexed =
      pcl::rvv_load::indexed_load_field_f32m2<PointT, Field>(base_u8, offsets, vl);
  pcl::rvv_store::strided_store_field_f32m2<PointT, Field>(base_u8, strided, vl);
  pcl::rvv_store::scatter_store_field_f32m2<PointT, Field>(
      base_u8, offsets, indexed, vl);
}

[[maybe_unused]] void
instantiate_all_field_tag_apis(std::uint8_t* base_u8,
                               const std::uint32_t* index_data,
                               const std::size_t vl)
{
  const vuint32m2_t indices = __riscv_vle32_v_u32m2(index_data, vl);
  instantiate_field_tag_api<pcl::PointXY, pcl::fields::x>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXY, pcl::fields::y>(base_u8, indices, vl);

  instantiate_field_tag_api<pcl::PointXYZI, pcl::fields::x>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZI, pcl::fields::y>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZI, pcl::fields::z>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZI, pcl::fields::intensity>(
      base_u8, indices, vl);

  instantiate_field_tag_api<pcl::PointXYZINormal, pcl::fields::x>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZINormal, pcl::fields::intensity>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZINormal, pcl::fields::normal_x>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZINormal, pcl::fields::normal_y>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZINormal, pcl::fields::normal_z>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointXYZINormal, pcl::fields::curvature>(
      base_u8, indices, vl);

  instantiate_field_tag_api<pcl::Normal, pcl::fields::normal_x>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::Normal, pcl::fields::normal_y>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::Normal, pcl::fields::normal_z>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::Normal, pcl::fields::curvature>(base_u8, indices, vl);

  instantiate_field_tag_api<pcl::PointNormal, pcl::fields::x>(base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointNormal, pcl::fields::normal_x>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointNormal, pcl::fields::normal_y>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointNormal, pcl::fields::normal_z>(
      base_u8, indices, vl);
  instantiate_field_tag_api<pcl::PointNormal, pcl::fields::curvature>(
      base_u8, indices, vl);
}

// Instantiate each public load family directly once. The runtime tests use the
// dispatch helpers; these calls keep the lower-level segment/fields primitives
// in the compile contract as well.
[[maybe_unused]] void
instantiate_all_load_apis(const std::uint8_t* base_u8,
                          const float* contiguous,
                          const std::uint32_t* index_data,
                          const std::size_t vl)
{
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);
  const vuint32m2_t indices = __riscv_vle32_v_u32m2(index_data, vl);
  const vuint32m2_t offsets =
      pcl::rvv_load::byte_offsets_u32m2<pcl::PointXYZI>(indices, vl);
  vfloat32m2_t vx, vy, vz;

  (void)pcl::rvv_load::gather_load_f32m2<pcl::PointXYZI, kX>(base_u8, offsets, vl);
  (void)pcl::rvv_load::strided_load_f32m2<sizeof(pcl::PointXYZI)>(
      reinterpret_cast<const float*>(base_u8 + kX), vl);
  pcl::rvv_load::strided_load3_seg_f32m2<sizeof(pcl::PointXYZI)>(
      reinterpret_cast<const float*>(base_u8 + kX), vl, vx, vy, vz);
  pcl::rvv_load::strided_load3_fields_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ>(
      base_u8, vl, vx, vy, vz);
  pcl::rvv_load::indexed_load3_seg_f32m2(
      reinterpret_cast<const float*>(base_u8 + kX), offsets, vl, vx, vy, vz);
  pcl::rvv_load::indexed_load3_fields_f32m2<pcl::PointXYZI, kX, kY, kZ>(
      base_u8, offsets, vl, vx, vy, vz);
  pcl::rvv_load::contiguous_seg3_load_f32m2(contiguous, vl, vx, vy, vz);
  pcl::rvv_load::indexed_load3_f32m2<pcl::PointXYZI, kX, kY, kZ>(
      base_u8, offsets, vl, vx, vy, vz);
  pcl::rvv_load::strided_load3_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ>(
      base_u8, vl, vx, vy, vz);
}

// Compile-only coverage for public store families that need no point-type
// semantics at runtime. Runtime tests below separately check their memory
// results, including tail and mask behavior.
[[maybe_unused]] void
instantiate_all_store_apis(std::uint8_t* base_u8,
                           float* contiguous,
                           const std::uint32_t* index_data,
                           const std::size_t vl)
{
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);
  constexpr std::size_t kI = offsetof(pcl::PointXYZI, intensity);
  const vuint32m2_t indices = __riscv_vle32_v_u32m2(index_data, vl);
  const vuint32m2_t offsets =
      pcl::rvv_store::byte_offsets_u32m2<pcl::PointXYZI>(indices, vl);
  const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(1.0f, vl);
  const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(2.0f, vl);
  const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(3.0f, vl);
  const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(4.0f, vl);
  const vbool16_t mask = __riscv_vmsne_vx_u32m2_b16(indices, 0u, vl);

  pcl::rvv_store::strided_store_f32m2<sizeof(pcl::PointXYZI)>(
      reinterpret_cast<float*>(base_u8 + kX), v0, vl);
  pcl::rvv_store::masked_strided_store_f32m2<sizeof(pcl::PointXYZI)>(
      mask, reinterpret_cast<float*>(base_u8 + kY), v1, vl);
  pcl::rvv_store::strided_store_field_f32m2<pcl::PointXYZI, pcl::fields::intensity>(
      base_u8, v3, vl);

  pcl::rvv_store::strided_store3_seg_f32m2<sizeof(pcl::PointXYZI)>(
      reinterpret_cast<float*>(base_u8 + kX), vl, v0, v1, v2);
  pcl::rvv_store::strided_store3_fields_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ>(
      base_u8, vl, v0, v1, v2);
  pcl::rvv_store::strided_store3_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ>(
      base_u8, vl, v0, v1, v2);
  pcl::rvv_store::strided_store4_seg_f32m2<sizeof(pcl::PointXYZI)>(
      reinterpret_cast<float*>(base_u8 + kX), vl, v0, v1, v2, v3);
  pcl::rvv_store::strided_store4_fields_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ, kI>(
      base_u8, vl, v0, v1, v2, v3);
  pcl::rvv_store::strided_store4_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ, kI>(
      base_u8, vl, v0, v1, v2, v3);

  pcl::rvv_store::contiguous_store3_f32m2(
      contiguous, contiguous, contiguous, vl, v0, v1, v2);
  pcl::rvv_store::contiguous_seg3_store_f32m2(contiguous, vl, v0, v1, v2);
  pcl::rvv_store::contiguous_store4_f32m2(
      contiguous, contiguous, contiguous, contiguous, vl, v0, v1, v2, v3);
  pcl::rvv_store::contiguous_seg4_store_f32m2(contiguous, vl, v0, v1, v2, v3);

  pcl::rvv_store::scatter_store_f32m2<kX>(base_u8, offsets, v0, vl);
  pcl::rvv_store::scatter_store_field_f32m2<pcl::PointXYZI, pcl::fields::intensity>(
      base_u8, offsets, v3, vl);
  pcl::rvv_store::scatter_store3_seg_f32m2(
      reinterpret_cast<float*>(base_u8 + kX), offsets, vl, v0, v1, v2);
  pcl::rvv_store::scatter_store3_fields_f32m2<kX, kY, kZ>(
      base_u8, offsets, vl, v0, v1, v2);
  pcl::rvv_store::scatter_store3_f32m2<kX, kY, kZ>(base_u8, offsets, vl, v0, v1, v2);
  pcl::rvv_store::scatter_store4_seg_f32m2(
      reinterpret_cast<float*>(base_u8 + kX), offsets, vl, v0, v1, v2, v3);
  pcl::rvv_store::scatter_store4_fields_f32m2<kX, kY, kZ, kI>(
      base_u8, offsets, vl, v0, v1, v2, v3);
  pcl::rvv_store::scatter_store4_f32m2<kX, kY, kZ, kI>(
      base_u8, offsets, vl, v0, v1, v2, v3);
}

} // namespace

TEST(RVVPointLoadStore, StridedThreeFieldLoadReadsEveryLane)
{
  constexpr std::size_t count = 64;
  auto points = makePoints(count);
  const auto* base = reinterpret_cast<const std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ>(
        base + i * sizeof(pcl::PointXYZI), vl, vx, vy, vz);

    const auto xs = readVector(vx, vl);
    const auto ys = readVector(vy, vl);
    const auto zs = readVector(vz, vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      EXPECT_FLOAT_EQ(xs[lane], points[i + lane].x);
      EXPECT_FLOAT_EQ(ys[lane], points[i + lane].y);
      EXPECT_FLOAT_EQ(zs[lane], points[i + lane].z);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, IndexedThreeFieldLoadReadsEveryLane)
{
  constexpr std::size_t count = 37;
  auto points = makePoints(count);
  const auto* base = reinterpret_cast<const std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    // Reverse the lane order so a helper that ignores the index vector fails.
    const vuint32m2_t local_reversed = __riscv_vrsub_vx_u32m2(
        __riscv_vid_v_u32m2(vl), static_cast<std::uint32_t>(vl - 1), vl);
    const vuint32m2_t indices =
        __riscv_vadd_vx_u32m2(local_reversed, static_cast<std::uint32_t>(i), vl);
    const vuint32m2_t offsets =
        pcl::rvv_load::byte_offsets_u32m2<pcl::PointXYZI>(indices, vl);

    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<pcl::PointXYZI, kX, kY, kZ>(
        base, offsets, vl, vx, vy, vz);

    const auto xs = readVector(vx, vl);
    const auto ys = readVector(vy, vl);
    const auto zs = readVector(vz, vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t source = i + vl - 1 - lane;
      EXPECT_FLOAT_EQ(xs[lane], points[source].x);
      EXPECT_FLOAT_EQ(ys[lane], points[source].y);
      EXPECT_FLOAT_EQ(zs[lane], points[source].z);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, IndexedThreeFieldScatterStoreUsesPermutationAndTail)
{
  constexpr std::size_t count = 37;
  auto points = makePoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vuint32m2_t local_reversed = __riscv_vrsub_vx_u32m2(
        __riscv_vid_v_u32m2(vl), static_cast<std::uint32_t>(vl - 1), vl);
    const vuint32m2_t indices =
        __riscv_vadd_vx_u32m2(local_reversed, static_cast<std::uint32_t>(i), vl);
    const vuint32m2_t offsets =
        pcl::rvv_store::byte_offsets_u32m2<pcl::PointXYZI>(indices, vl);

    pcl::rvv_store::scatter_store3_f32m2<kX, kY, kZ>(
        base,
        offsets,
        vl,
        __riscv_vfmv_v_f_f32m2(1000.0f, vl),
        __riscv_vfmv_v_f_f32m2(2000.0f, vl),
        __riscv_vfmv_v_f_f32m2(3000.0f, vl));

    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t point = i + vl - 1 - lane;
      EXPECT_FLOAT_EQ(points[point].x, 1000.0f);
      EXPECT_FLOAT_EQ(points[point].y, 2000.0f);
      EXPECT_FLOAT_EQ(points[point].z, 3000.0f);
      EXPECT_FLOAT_EQ(points[point].intensity, static_cast<float>(point) + 0.75f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, IndexedNormalFieldsUsePermutationAndTail)
{
  constexpr std::size_t count = 37;
  auto points = makeNormalPoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());
  constexpr std::size_t kNX = offsetof(pcl::PointXYZINormal, normal_x);
  constexpr std::size_t kNY = offsetof(pcl::PointXYZINormal, normal_y);
  constexpr std::size_t kNZ = offsetof(pcl::PointXYZINormal, normal_z);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vuint32m2_t local_reversed = __riscv_vrsub_vx_u32m2(
        __riscv_vid_v_u32m2(vl), static_cast<std::uint32_t>(vl - 1), vl);
    const vuint32m2_t indices =
        __riscv_vadd_vx_u32m2(local_reversed, static_cast<std::uint32_t>(i), vl);
    const vuint32m2_t offsets =
        pcl::rvv_load::byte_offsets_u32m2<pcl::PointXYZINormal>(indices, vl);

    vfloat32m2_t normal_x, normal_y, normal_z;
    pcl::rvv_load::indexed_load3_f32m2<pcl::PointXYZINormal, kNX, kNY, kNZ>(
        base, offsets, vl, normal_x, normal_y, normal_z);
    const auto xs = readVector(normal_x, vl);
    const auto ys = readVector(normal_y, vl);
    const auto zs = readVector(normal_z, vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t point = i + vl - 1 - lane;
      EXPECT_FLOAT_EQ(xs[lane], static_cast<float>(point) + 1.1f);
      EXPECT_FLOAT_EQ(ys[lane], static_cast<float>(point) + 1.2f);
      EXPECT_FLOAT_EQ(zs[lane], static_cast<float>(point) + 1.3f);
    }

    pcl::rvv_store::scatter_store3_f32m2<kNX, kNY, kNZ>(
        base,
        offsets,
        vl,
        __riscv_vfmv_v_f_f32m2(4000.0f, vl),
        __riscv_vfmv_v_f_f32m2(5000.0f, vl),
        __riscv_vfmv_v_f_f32m2(6000.0f, vl));
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t point = i + vl - 1 - lane;
      EXPECT_FLOAT_EQ(points[point].normal_x, 4000.0f);
      EXPECT_FLOAT_EQ(points[point].normal_y, 5000.0f);
      EXPECT_FLOAT_EQ(points[point].normal_z, 6000.0f);
      EXPECT_FLOAT_EQ(points[point].x, static_cast<float>(point) + 0.1f);
      EXPECT_FLOAT_EQ(points[point].curvature, static_cast<float>(point) + 1.4f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, StridedThreeFieldStoreWritesEveryLane)
{
  constexpr std::size_t count = 64;
  auto points = makePoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vfloat32m2_t vx = __riscv_vfmv_v_f_f32m2(1000.0f, vl);
    const vfloat32m2_t vy = __riscv_vfmv_v_f_f32m2(2000.0f, vl);
    const vfloat32m2_t vz = __riscv_vfmv_v_f_f32m2(3000.0f, vl);

    pcl::rvv_store::strided_store3_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ>(
        base + i * sizeof(pcl::PointXYZI), vl, vx, vy, vz);

    for (std::size_t lane = 0; lane < vl; ++lane) {
      EXPECT_FLOAT_EQ(points[i + lane].x, 1000.0f);
      EXPECT_FLOAT_EQ(points[i + lane].y, 2000.0f);
      EXPECT_FLOAT_EQ(points[i + lane].z, 3000.0f);
      EXPECT_FLOAT_EQ(points[i + lane].intensity, static_cast<float>(i + lane) + 0.75f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, NamedFloatFieldRoundTrips)
{
  constexpr std::size_t count = 64;
  auto points = makePoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    auto* chunk = base + i * sizeof(pcl::PointXYZI);

    const vfloat32m2_t loaded =
        pcl::rvv_load::strided_load_field_f32m2<pcl::PointXYZI, pcl::fields::intensity>(
            chunk, vl);
    pcl::rvv_store::strided_store_field_f32m2<pcl::PointXYZI, pcl::fields::intensity>(
        chunk, __riscv_vfadd_vf_f32m2(loaded, 1.0f, vl), vl);

    const auto values = readVector(
        pcl::rvv_load::strided_load_field_f32m2<pcl::PointXYZI, pcl::fields::intensity>(
            chunk, vl),
        vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      EXPECT_FLOAT_EQ(values[lane], static_cast<float>(i + lane) + 1.75f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, NamedNormalFieldStridedLoadAndStoreUseFieldTags)
{
  constexpr std::size_t count = 37;
  auto points = makeNormalPoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    auto* chunk = base + i * sizeof(pcl::PointXYZINormal);
    const vfloat32m2_t normal_y =
        pcl::rvv_load::strided_load_field_f32m2<pcl::PointXYZINormal,
                                                pcl::fields::normal_y>(chunk, vl);
    pcl::rvv_store::strided_store_field_f32m2<pcl::PointXYZINormal,
                                              pcl::fields::normal_y>(
        chunk, __riscv_vfadd_vf_f32m2(normal_y, 10.0f, vl), vl);

    const auto values = readVector(
        pcl::rvv_load::strided_load_field_f32m2<pcl::PointXYZINormal,
                                                pcl::fields::normal_y>(chunk, vl),
        vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t point = i + lane;
      EXPECT_FLOAT_EQ(values[lane], static_cast<float>(point) + 11.2f);
      EXPECT_FLOAT_EQ(points[point].normal_x, static_cast<float>(point) + 1.1f);
      EXPECT_FLOAT_EQ(points[point].curvature, static_cast<float>(point) + 1.4f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, NamedFieldIndexedLoadAndScatterStoreUsePointOffsets)
{
  constexpr std::size_t count = 37;
  auto points = makeNormalPoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vuint32m2_t local_reversed = __riscv_vrsub_vx_u32m2(
        __riscv_vid_v_u32m2(vl), static_cast<std::uint32_t>(vl - 1), vl);
    const vuint32m2_t indices =
        __riscv_vadd_vx_u32m2(local_reversed, static_cast<std::uint32_t>(i), vl);
    const vuint32m2_t offsets =
        pcl::rvv_load::byte_offsets_u32m2<pcl::PointXYZINormal>(indices, vl);

    const vfloat32m2_t loaded =
        pcl::rvv_load::indexed_load_field_f32m2<pcl::PointXYZINormal,
                                                pcl::fields::curvature>(
            base, offsets, vl);
    pcl::rvv_store::scatter_store_field_f32m2<pcl::PointXYZINormal,
                                              pcl::fields::curvature>(
        base, offsets, __riscv_vfadd_vf_f32m2(loaded, 20.0f, vl), vl);

    const auto values =
        readVector(pcl::rvv_load::indexed_load_field_f32m2<pcl::PointXYZINormal,
                                                           pcl::fields::curvature>(
                       base, offsets, vl),
                   vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t point = i + vl - 1 - lane;
      EXPECT_FLOAT_EQ(values[lane], static_cast<float>(point) + 21.4f);
      EXPECT_FLOAT_EQ(points[point].normal_x, static_cast<float>(point) + 1.1f);
      EXPECT_FLOAT_EQ(points[point].normal_y, static_cast<float>(point) + 1.2f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, StridedFourFieldStoreWritesEveryField)
{
  constexpr std::size_t count = 37;
  auto points = makePoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);
  constexpr std::size_t kI = offsetof(pcl::PointXYZI, intensity);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    pcl::rvv_store::strided_store4_f32m2<sizeof(pcl::PointXYZI), kX, kY, kZ, kI>(
        base + i * sizeof(pcl::PointXYZI),
        vl,
        __riscv_vfmv_v_f_f32m2(1000.0f, vl),
        __riscv_vfmv_v_f_f32m2(2000.0f, vl),
        __riscv_vfmv_v_f_f32m2(3000.0f, vl),
        __riscv_vfmv_v_f_f32m2(4000.0f, vl));
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const auto& point = points[i + lane];
      EXPECT_FLOAT_EQ(point.x, 1000.0f);
      EXPECT_FLOAT_EQ(point.y, 2000.0f);
      EXPECT_FLOAT_EQ(point.z, 3000.0f);
      EXPECT_FLOAT_EQ(point.intensity, 4000.0f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, IndexedFourFieldScatterStoreUsesPermutationAndTail)
{
  constexpr std::size_t count = 37;
  auto points = makePoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);
  constexpr std::size_t kY = offsetof(pcl::PointXYZI, y);
  constexpr std::size_t kZ = offsetof(pcl::PointXYZI, z);
  constexpr std::size_t kI = offsetof(pcl::PointXYZI, intensity);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vuint32m2_t local_reversed = __riscv_vrsub_vx_u32m2(
        __riscv_vid_v_u32m2(vl), static_cast<std::uint32_t>(vl - 1), vl);
    const vuint32m2_t indices =
        __riscv_vadd_vx_u32m2(local_reversed, static_cast<std::uint32_t>(i), vl);
    const vuint32m2_t offsets =
        pcl::rvv_store::byte_offsets_u32m2<pcl::PointXYZI>(indices, vl);
    pcl::rvv_store::scatter_store4_f32m2<kX, kY, kZ, kI>(
        base,
        offsets,
        vl,
        __riscv_vfmv_v_f_f32m2(5000.0f, vl),
        __riscv_vfmv_v_f_f32m2(6000.0f, vl),
        __riscv_vfmv_v_f_f32m2(7000.0f, vl),
        __riscv_vfmv_v_f_f32m2(8000.0f, vl));
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const auto& point = points[i + vl - 1 - lane];
      EXPECT_FLOAT_EQ(point.x, 5000.0f);
      EXPECT_FLOAT_EQ(point.y, 6000.0f);
      EXPECT_FLOAT_EQ(point.z, 7000.0f);
      EXPECT_FLOAT_EQ(point.intensity, 8000.0f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, ContiguousStoreFamiliesWriteThreeAndFourFields)
{
  constexpr std::size_t count = 37;
  std::vector<float> plain3_0(count, -1.0f);
  std::vector<float> plain3_1(count, -1.0f);
  std::vector<float> plain3_2(count, -1.0f);
  std::vector<float> plain4_0(count, -1.0f);
  std::vector<float> plain4_1(count, -1.0f);
  std::vector<float> plain4_2(count, -1.0f);
  std::vector<float> plain4_3(count, -1.0f);
  std::vector<float> packed3(count * 3, -1.0f);
  std::vector<float> packed4(count * 4, -1.0f);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vfloat32m2_t v0 = __riscv_vfmv_v_f_f32m2(10.0f, vl);
    const vfloat32m2_t v1 = __riscv_vfmv_v_f_f32m2(20.0f, vl);
    const vfloat32m2_t v2 = __riscv_vfmv_v_f_f32m2(30.0f, vl);
    const vfloat32m2_t v3 = __riscv_vfmv_v_f_f32m2(40.0f, vl);
    pcl::rvv_store::contiguous_store3_f32m2(
        plain3_0.data() + i, plain3_1.data() + i, plain3_2.data() + i, vl, v0, v1, v2);
    pcl::rvv_store::contiguous_seg3_store_f32m2(packed3.data() + i * 3, vl, v0, v1, v2);
    pcl::rvv_store::contiguous_store4_f32m2(plain4_0.data() + i,
                                            plain4_1.data() + i,
                                            plain4_2.data() + i,
                                            plain4_3.data() + i,
                                            vl,
                                            v0,
                                            v1,
                                            v2,
                                            v3);
    pcl::rvv_store::contiguous_seg4_store_f32m2(
        packed4.data() + i * 4, vl, v0, v1, v2, v3);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const std::size_t point = i + lane;
      EXPECT_FLOAT_EQ(plain3_0[point], 10.0f);
      EXPECT_FLOAT_EQ(plain3_1[point], 20.0f);
      EXPECT_FLOAT_EQ(plain3_2[point], 30.0f);
      EXPECT_FLOAT_EQ(plain4_0[point], 10.0f);
      EXPECT_FLOAT_EQ(plain4_1[point], 20.0f);
      EXPECT_FLOAT_EQ(plain4_2[point], 30.0f);
      EXPECT_FLOAT_EQ(plain4_3[point], 40.0f);
      EXPECT_FLOAT_EQ(packed3[point * 3 + 0], 10.0f);
      EXPECT_FLOAT_EQ(packed3[point * 3 + 1], 20.0f);
      EXPECT_FLOAT_EQ(packed3[point * 3 + 2], 30.0f);
      EXPECT_FLOAT_EQ(packed4[point * 4 + 0], 10.0f);
      EXPECT_FLOAT_EQ(packed4[point * 4 + 1], 20.0f);
      EXPECT_FLOAT_EQ(packed4[point * 4 + 2], 30.0f);
      EXPECT_FLOAT_EQ(packed4[point * 4 + 3], 40.0f);
    }
    i += vl;
  }
}

TEST(RVVPointLoadStore, MaskedStridedStorePreservesUnselectedLanes)
{
  constexpr std::size_t count = 37;
  auto points = makePoints(count);
  auto* base = reinterpret_cast<std::uint8_t*>(points.data());
  constexpr std::size_t kX = offsetof(pcl::PointXYZI, x);

  for (std::size_t i = 0; i < count;) {
    const std::size_t vl = __riscv_vsetvl_e32m2(count - i);
    const vuint32m2_t lane_ids = __riscv_vid_v_u32m2(vl);
    const vbool16_t odd_lanes =
        __riscv_vmsne_vx_u32m2_b16(__riscv_vand_vx_u32m2(lane_ids, 1u, vl), 0u, vl);
    pcl::rvv_store::masked_strided_store_f32m2<sizeof(pcl::PointXYZI)>(
        odd_lanes,
        reinterpret_cast<float*>(base + i * sizeof(pcl::PointXYZI) + kX),
        __riscv_vfmv_v_f_f32m2(9000.0f, vl),
        vl);
    for (std::size_t lane = 0; lane < vl; ++lane) {
      const auto& point = points[i + lane];
      EXPECT_FLOAT_EQ(point.x, lane % 2 == 1 ? 9000.0f : static_cast<float>(i + lane));
      EXPECT_FLOAT_EQ(point.y, static_cast<float>(i + lane) + 0.25f);
      EXPECT_FLOAT_EQ(point.z, static_cast<float>(i + lane) + 0.5f);
      EXPECT_FLOAT_EQ(point.intensity, static_cast<float>(i + lane) + 0.75f);
    }
    i += vl;
  }
}

#endif // __RVV10__
