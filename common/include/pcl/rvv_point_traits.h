/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2026-, Open Perception Inc.
 *
 *  All rights reserved
 */

#pragma once

#include <pcl/field_traits.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace pcl {
namespace rvv {

/** \brief Strip cv/ref qualifiers from a field expression type.
 *
 * Member-field gates use expressions such as \c decltype(std::declval<PointT>().x).
 * This helper normalizes those expression types before comparing them with
 * \c float. It intentionally does not inspect PCL field registration metadata.
 */
template <typename T>
using RVVFieldScalar = std::remove_cv_t<std::remove_reference_t<T>>;

/** \brief True when a PCL-registered field is exactly one \c float scalar.
 *
 * This is a field-semantics gate: it uses \c pcl::traits::has_field and
 * \c pcl::traits::datatype, so it covers registered PCL point fields rather
 * than direct C++ member expressions. It does not check POD layout, byte
 * offsets, alignment, or algorithm-specific dispatch conditions.
 */
template <typename PointT,
          typename Field,
          bool HasField = pcl::traits::has_field<PointT, Field>::value>
struct RVVFloatFieldLayout : std::false_type {};

template <typename PointT, typename Field>
struct RVVFloatFieldLayout<PointT, Field, true>
: std::bool_constant<
      std::is_same_v<typename pcl::traits::datatype<PointT, Field>::decomposed::type,
                     float> &&
      pcl::traits::datatype<PointT, Field>::decomposed::value == 1> {};

/** \brief PCL traits gate for point types with registered single-float x/y/z.
 *
 * Use this when an RVV algorithm needs the semantic guarantee that the current
 * \c PointT has registered \c x, \c y, and \c z fields, each represented as a
 * single \c float. The exposed offsets are the offsets for the current
 * \c PointT; source and target point types must therefore be gated separately.
 *
 * This gate deliberately does not require \c PointT or its POD type to be
 * standard-layout. Callers that directly reinterpret AoS byte offsets must
 * rely on the load/store helper static_asserts or add the stronger local gate
 * required by their access pattern.
 */
template <typename PointT, bool HasXYZ = pcl::traits::has_xyz<PointT>::value>
struct RVVXYZFloatLayout : std::false_type {};

template <typename PointT>
struct RVVXYZFloatLayout<PointT, true>
: std::bool_constant<RVVFloatFieldLayout<PointT, pcl::fields::x>::value &&
                     RVVFloatFieldLayout<PointT, pcl::fields::y>::value &&
                     RVVFloatFieldLayout<PointT, pcl::fields::z>::value> {
  static constexpr std::size_t kX = pcl::traits::offset<PointT, pcl::fields::x>::value;
  static constexpr std::size_t kY = pcl::traits::offset<PointT, pcl::fields::y>::value;
  static constexpr std::size_t kZ = pcl::traits::offset<PointT, pcl::fields::z>::value;
};

/** \brief Strong AoS layout gate for registered single-float xyz fields.
 *
 * This is for algorithms that directly read \c x/y/z from an AoS point cloud
 * using byte offsets. Besides the field-semantics checks, it verifies the POD
 * standard-layout assumption, that \c PointT and its POD representation have
 * the same size, and that the stride and field offsets are aligned for
 * \c float access.
 *
 * It is still only a point layout gate: size thresholds, index/correspondence
 * overload policy, Scalar type policy, and output-order guarantees remain
 * algorithm dispatch/fallback decisions.
 */
template <typename PointT, bool HasXYZ = pcl::traits::has_xyz<PointT>::value>
struct RVVXYZAoSFloatLayout : std::false_type {};

template <typename PointT>
struct RVVXYZAoSFloatLayout<PointT, true> {
  using Pod = typename pcl::traits::POD<PointT>::type;

  static constexpr std::size_t kX = RVVXYZFloatLayout<PointT>::kX;
  static constexpr std::size_t kY = RVVXYZFloatLayout<PointT>::kY;
  static constexpr std::size_t kZ = RVVXYZFloatLayout<PointT>::kZ;

  static constexpr bool value =
      RVVXYZFloatLayout<PointT>::value && std::is_standard_layout_v<Pod> &&
      sizeof(PointT) == sizeof(Pod) && sizeof(PointT) % alignof(float) == 0 &&
      kX % alignof(float) == 0 && kY % alignof(float) == 0 && kZ % alignof(float) == 0;
};

/** \brief PCL traits gate for point types with registered single-float normal fields.
 *
 * This is a field-semantics gate for normal clouds. It verifies registered
 * \c normal_x, \c normal_y, and \c normal_z fields, each represented as one
 * \c float, and exposes their PCL traits offsets. It does not imply that the
 * same point type also has XYZ coordinates or curvature.
 */
template <typename PointT, bool HasNormal = pcl::traits::has_normal<PointT>::value>
struct RVVNormalFloatLayout : std::false_type {};

template <typename PointT>
struct RVVNormalFloatLayout<PointT, true>
: std::bool_constant<RVVFloatFieldLayout<PointT, pcl::fields::normal_x>::value &&
                     RVVFloatFieldLayout<PointT, pcl::fields::normal_y>::value &&
                     RVVFloatFieldLayout<PointT, pcl::fields::normal_z>::value> {
  static constexpr std::size_t kNormalX =
      pcl::traits::offset<PointT, pcl::fields::normal_x>::value;
  static constexpr std::size_t kNormalY =
      pcl::traits::offset<PointT, pcl::fields::normal_y>::value;
  static constexpr std::size_t kNormalZ =
      pcl::traits::offset<PointT, pcl::fields::normal_z>::value;
};

/** \brief Strong AoS layout gate for registered single-float xyz + normal fields.
 *
 * This is for algorithms that directly read \c x/y/z and
 * \c normal_x/normal_y/normal_z from an AoS point cloud using byte offsets.
 * Besides the field-semantics checks, it verifies the POD standard-layout
 * assumption, that \c PointT and its POD representation have the same size,
 * and that the stride and field offsets are aligned for \c float access.
 *
 * It is still only a point layout gate: size thresholds, VLEN scratch-buffer
 * limits, index/correspondence overload policy, Scalar type policy, and
 * output-order guarantees remain algorithm dispatch/fallback decisions.
 */
template <typename PointT,
          bool HasXYZ = pcl::traits::has_xyz<PointT>::value,
          bool HasNormal = pcl::traits::has_normal<PointT>::value>
struct RVVXYZNormalFloatLayout : std::false_type {};

template <typename PointT>
struct RVVXYZNormalFloatLayout<PointT, true, true> {
  using Pod = typename pcl::traits::POD<PointT>::type;

  static constexpr std::size_t kX = pcl::traits::offset<PointT, pcl::fields::x>::value;
  static constexpr std::size_t kY = pcl::traits::offset<PointT, pcl::fields::y>::value;
  static constexpr std::size_t kZ = pcl::traits::offset<PointT, pcl::fields::z>::value;
  static constexpr std::size_t kNX =
      pcl::traits::offset<PointT, pcl::fields::normal_x>::value;
  static constexpr std::size_t kNY =
      pcl::traits::offset<PointT, pcl::fields::normal_y>::value;
  static constexpr std::size_t kNZ =
      pcl::traits::offset<PointT, pcl::fields::normal_z>::value;

  static constexpr bool value =
      RVVXYZAoSFloatLayout<PointT>::value &&
      RVVFloatFieldLayout<PointT, pcl::fields::normal_x>::value &&
      RVVFloatFieldLayout<PointT, pcl::fields::normal_y>::value &&
      RVVFloatFieldLayout<PointT, pcl::fields::normal_z>::value &&
      std::is_standard_layout_v<Pod> && sizeof(PointT) == sizeof(Pod) &&
      sizeof(PointT) % alignof(float) == 0 && kX % alignof(float) == 0 &&
      kY % alignof(float) == 0 && kZ % alignof(float) == 0 &&
      kNX % alignof(float) == 0 && kNY % alignof(float) == 0 &&
      kNZ % alignof(float) == 0;
};

namespace detail {

/** \brief Legacy member-expression gate used by common load/store call sites.
 *
 * This checks direct C++ members named \c x, \c y, and \c z and requires those
 * member expression types to be \c float on a standard-layout \c PointT. It is
 * intentionally different from the public \c pcl::rvv::RVVXYZFloatLayout:
 * it does not require PCL field registration, and it does not expose PCL
 * traits offsets.
 */
template <typename PointT, typename = void>
struct RVVXYZFloatLayout : std::false_type {};

template <typename PointT>
struct RVVXYZFloatLayout<PointT,
                         std::void_t<decltype(std::declval<PointT>().x),
                                     decltype(std::declval<PointT>().y),
                                     decltype(std::declval<PointT>().z)>>
: std::bool_constant<
      std::is_standard_layout_v<PointT> &&
      std::is_same_v<RVVFieldScalar<decltype(std::declval<PointT>().x)>, float> &&
      std::is_same_v<RVVFieldScalar<decltype(std::declval<PointT>().y)>, float> &&
      std::is_same_v<RVVFieldScalar<decltype(std::declval<PointT>().z)>, float>> {};

} // namespace detail

/** \brief Compatibility predicate for legacy member x/y/z RVV load/store gates.
 *
 * Prefer this when migrating existing local gates that are exactly:
 * standard-layout \c PointT plus direct member \c x/y/z of type \c float.
 * Do not use it for Z-only gates, exact-type specializations such as
 * \c PointXYZ-only paths, or algorithms that need PCL-registered field
 * semantics.
 */
template <typename PointT>
inline constexpr bool kRVVXYZPointCompatible = detail::RVVXYZFloatLayout<PointT>::value;

/** \brief Variable-template form of \c RVVXYZAoSFloatLayout<PointT>::value. */
template <typename PointT>
inline constexpr bool kRVVXYZAoSPointCompatible = RVVXYZAoSFloatLayout<PointT>::value;

/** \brief Variable-template form of \c RVVXYZNormalFloatLayout<PointT>::value. */
template <typename PointT>
inline constexpr bool kRVVXYZNormalPointCompatible =
    RVVXYZNormalFloatLayout<PointT>::value;

/** \brief Maximum point count representable by 32-bit byte offsets.
 *
 * RVV indexed gather/scatter helpers in this topic use byte offsets stored in
 * \c uint32_t vectors. If all indices are already known to be valid for a
 * cloud, proving
 * \code
 * cloud.size() <= rvvMaxU32ByteOffsetElements<PointT>()
 * \endcode
 * is enough to show that every legal \c index * sizeof(PointT) byte offset is
 * representable. This helper does not validate raw index contents; callers
 * with untrusted indices must check those separately or fall back to scalar.
 */
template <typename PointT>
constexpr std::size_t
rvvMaxU32ByteOffsetElements()
{
  return static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() /
                                  sizeof(PointT));
}

} // namespace rvv
} // namespace pcl
