/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-present, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
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
 *
 * $Id$
 *
 */

#pragma once

#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/common/point_tests.h> // for pcl::isFinite
#include <Eigen/Eigenvalues> // for EigenSolver

#include <boost/fusion/algorithm/transformation/filter_if.hpp> // for boost::fusion::filter_if
#include <boost/fusion/algorithm/iteration/for_each.hpp> // for boost::fusion::for_each
#include <boost/mpl/size.hpp> // for boost::mpl::size

#include <type_traits> // for std::is_same_v (Eigen demeanPointCloud dispatch)

#if defined(__RVV10__)
#include <riscv_vector.h>
#include <cstdint>
#include <cstddef>
#include <type_traits>

#include <pcl/common/rvv_point_load.h>
#include <pcl/common/rvv_point_store.h>
#endif

namespace pcl
{

template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroid (ConstCloudIterator<PointT> &cloud_iterator,
                   Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  Eigen::Matrix<Scalar, 4, 1> accumulator {0, 0, 0, 0};

  unsigned int cp = 0;

  // For each point in the cloud
  // If the data is dense, we don't need to check for NaN
  while (cloud_iterator.isValid ())
  {
    // Check if the point is invalid
    if (pcl::isFinite (*cloud_iterator))
    {
      accumulator[0] += cloud_iterator->x;
      accumulator[1] += cloud_iterator->y;
      accumulator[2] += cloud_iterator->z;
      ++cp;
    }
    ++cloud_iterator;
  }

  if (cp > 0) {
    centroid = accumulator;
    centroid /= static_cast<Scalar> (cp);
    centroid[3] = 1;
  }
  return (cp);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroidStandard (const pcl::PointCloud<PointT> &cloud,
                           Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // Dense path only. Caller guarantees cloud not empty and cloud.is_dense == true.
  // Initialize to 0
  centroid.setZero ();
  for (const auto& point: cloud)
  {
    centroid[0] += point.x;
    centroid[1] += point.y;
    centroid[2] += point.z;
  }
  centroid /= static_cast<Scalar> (cloud.size ());
  centroid[3] = 1;
  return (static_cast<unsigned int> (cloud.size ()));
}

#if defined(__RVV10__)
template <typename PointT, typename Scalar>
inline unsigned int
compute3DCentroidRVV (const pcl::PointCloud<PointT> &cloud,
                      Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  const std::size_t n = cloud.size ();
  if (n < 16)
    return compute3DCentroidStandard (cloud, centroid);

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc_x = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc_y = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc_z = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto* base = reinterpret_cast<const std::uint8_t*> (cloud.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    v_acc_x = __riscv_vfadd_vv_f32m2_tu (v_acc_x, v_acc_x, vx, vl);
    v_acc_y = __riscv_vfadd_vv_f32m2_tu (v_acc_y, v_acc_y, vy, vl);
    v_acc_z = __riscv_vfadd_vv_f32m2_tu (v_acc_z, v_acc_z, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const Scalar sx = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_x, v_zero, vlmax)));
  const Scalar sy = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_y, v_zero, vlmax)));
  const Scalar sz = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_z, v_zero, vlmax)));

  const Scalar inv_n = static_cast<Scalar> (1.0) / static_cast<Scalar> (n);
  centroid[0] = sx * inv_n;
  centroid[1] = sy * inv_n;
  centroid[2] = sz * inv_n;
  centroid[3] = 1;
  return (static_cast<unsigned int> (n));
}
#endif

template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
                   Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  if (cloud.empty ())
    return (0);

  // For each point in the cloud
  // If the data is dense, we don't need to check for NaN
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return compute3DCentroidRVV (cloud, centroid);
#endif
    return compute3DCentroidStandard (cloud, centroid);
  }

  // NaN or Inf values could exist => check for them (keep scalar)
  unsigned int cp = 0;
  Eigen::Matrix<Scalar, 4, 1> accumulator {0, 0, 0, 0};
  for (const auto& point: cloud)
  {
    // Check if the point is invalid
    if (!isFinite (point))
      continue;

    accumulator[0] += point.x;
    accumulator[1] += point.y;
    accumulator[2] += point.z;
    ++cp;
  }
  if (cp > 0) {
    centroid = accumulator;
    centroid /= static_cast<Scalar> (cp);
    centroid[3] = 1;
  }

  return (cp);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroidStandard (const pcl::PointCloud<PointT> &cloud,
                           const Indices &indices,
                           Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // Dense path only; caller guarantees indices non-empty and cloud.is_dense == true
  // Initialize to 0
  centroid.setZero ();
  for (const auto& index : indices)
  {
    centroid[0] += cloud[index].x;
    centroid[1] += cloud[index].y;
    centroid[2] += cloud[index].z;
  }
  centroid /= static_cast<Scalar> (indices.size ());
  centroid[3] = 1;
  return (static_cast<unsigned int> (indices.size ()));
}

#if defined(__RVV10__)
template <typename PointT, typename Scalar>
inline unsigned int
compute3DCentroidRVV (const pcl::PointCloud<PointT> &cloud,
                      const Indices &indices,
                      Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  const std::size_t n = indices.size ();
  if (n < 16)
    return compute3DCentroidStandard (cloud, indices, centroid);

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc_x = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc_y = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc_z = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto* base = reinterpret_cast<const std::uint8_t*> (cloud.data ());
  const auto* idx_i32 = reinterpret_cast<const std::int32_t*> (indices.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base, v_off, vl, vx, vy, vz);
    v_acc_x = __riscv_vfadd_vv_f32m2_tu (v_acc_x, v_acc_x, vx, vl);
    v_acc_y = __riscv_vfadd_vv_f32m2_tu (v_acc_y, v_acc_y, vy, vl);
    v_acc_z = __riscv_vfadd_vv_f32m2_tu (v_acc_z, v_acc_z, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const Scalar sx = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_x, v_zero, vlmax)));
  const Scalar sy = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_y, v_zero, vlmax)));
  const Scalar sz = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_z, v_zero, vlmax)));
  const Scalar inv_n = static_cast<Scalar> (1.0) / static_cast<Scalar> (n);
  centroid[0] = sx * inv_n;
  centroid[1] = sy * inv_n;
  centroid[2] = sz * inv_n;
  centroid[3] = 1;
  return (static_cast<unsigned int> (n));
}
#endif

template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
                   const Indices &indices,
                   Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  if (indices.empty ())
    return (0);

  // If the data is dense, we don't need to check for NaN
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return compute3DCentroidRVV (cloud, indices, centroid);
#endif
    return compute3DCentroidStandard (cloud, indices, centroid);
  }
  // NaN or Inf values could exist => check for them
  Eigen::Matrix<Scalar, 4, 1> accumulator {0, 0, 0, 0};
  unsigned int cp = 0;
  for (const auto& index : indices)
  {
    // Check if the point is invalid
    if (!isFinite (cloud [index]))
      continue;

    accumulator[0] += cloud[index].x;
    accumulator[1] += cloud[index].y;
    accumulator[2] += cloud[index].z;
    ++cp;
  }
  if (cp > 0) {
    centroid = accumulator;
    centroid /= static_cast<Scalar> (cp);
    centroid[3] = 1;
  }
  return (cp);
}

template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
                   const pcl::PointIndices &indices,
                   Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  return (pcl::compute3DCentroid (cloud, indices.indices, centroid));
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixCentroidStandard (const pcl::PointCloud<PointT> &cloud,
                                         const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  covariance_matrix.setZero ();
  // For each point in the cloud
  for (const auto& point: cloud)
  {
    Eigen::Matrix<Scalar, 4, 1> pt;
    pt[0] = point.x - centroid[0];
    pt[1] = point.y - centroid[1];
    pt[2] = point.z - centroid[2];

    covariance_matrix (1, 1) += pt.y () * pt.y ();
    covariance_matrix (1, 2) += pt.y () * pt.z ();

    covariance_matrix (2, 2) += pt.z () * pt.z ();

    pt *= pt.x ();
    covariance_matrix (0, 0) += pt.x ();
    covariance_matrix (0, 1) += pt.y ();
    covariance_matrix (0, 2) += pt.z ();
  }
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
  covariance_matrix (2, 0) = covariance_matrix (0, 2);
  covariance_matrix (2, 1) = covariance_matrix (1, 2);
  return static_cast<unsigned int> (cloud.size ());
}

#if defined(__RVV10__)
/** Dense full cloud; \a n < 16 delegates to Standard (same pattern as \c compute3DCentroidRVV). */
template <typename PointT, typename Scalar>
inline unsigned int
computeCovarianceMatrixCentroidRVV (const pcl::PointCloud<PointT> &cloud,
                                    const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                    Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  const std::size_t n = cloud.size ();
  if (n < 16)
    return computeCovarianceMatrixCentroidStandard (cloud, centroid, covariance_matrix);

  const float cx = static_cast<float> (centroid[0]);
  const float cy = static_cast<float> (centroid[1]);
  const float cz = static_cast<float> (centroid[2]);

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc0 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc1 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc2 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc3 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc4 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc5 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto *base = reinterpret_cast<const std::uint8_t *> (cloud.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    // FMA + _tu: vd += vs1*vs2; tail undisturbed when vl < VLMAX (see doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const float s0 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc0, v_zero, vlmax));
  const float s1 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc1, v_zero, vlmax));
  const float s2 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc2, v_zero, vlmax));
  const float s3 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc3, v_zero, vlmax));
  const float s4 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc4, v_zero, vlmax));
  const float s5 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc5, v_zero, vlmax));

  covariance_matrix (0, 0) = static_cast<Scalar> (s0);
  covariance_matrix (0, 1) = static_cast<Scalar> (s1);
  covariance_matrix (0, 2) = static_cast<Scalar> (s2);
  covariance_matrix (1, 1) = static_cast<Scalar> (s3);
  covariance_matrix (1, 2) = static_cast<Scalar> (s4);
  covariance_matrix (2, 2) = static_cast<Scalar> (s5);
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
  covariance_matrix (2, 0) = covariance_matrix (0, 2);
  covariance_matrix (2, 1) = covariance_matrix (1, 2);
  return static_cast<unsigned int> (n);
}
#endif // __RVV10__

template <typename PointT, typename Scalar> inline unsigned
computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                         const Eigen::Matrix<Scalar, 4, 1> &centroid,
                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  if (cloud.empty ())
    return (0);

  unsigned point_count;
  // If the data is dense, we don't need to check for NaN
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return computeCovarianceMatrixCentroidRVV (cloud, centroid, covariance_matrix);
#endif
    return computeCovarianceMatrixCentroidStandard (cloud, centroid, covariance_matrix);
  }
  // NaN or Inf values could exist => check for them
  else
  {
    Eigen::Matrix<Scalar, 3, 3> temp_covariance_matrix;
    temp_covariance_matrix.setZero ();
    point_count = 0;
    // For each point in the cloud
    for (const auto& point: cloud)
    {
      // Check if the point is invalid
      if (!isFinite (point))
        continue;

      Eigen::Matrix<Scalar, 4, 1> pt;
      pt[0] = point.x - centroid[0];
      pt[1] = point.y - centroid[1];
      pt[2] = point.z - centroid[2];

      temp_covariance_matrix (1, 1) += pt.y () * pt.y ();
      temp_covariance_matrix (1, 2) += pt.y () * pt.z ();

      temp_covariance_matrix (2, 2) += pt.z () * pt.z ();

      pt *= pt.x ();
      temp_covariance_matrix (0, 0) += pt.x ();
      temp_covariance_matrix (0, 1) += pt.y ();
      temp_covariance_matrix (0, 2) += pt.z ();
      ++point_count;
    }
    if (point_count > 0) {
      covariance_matrix = temp_covariance_matrix;
    }
  }
  if (point_count == 0) {
    return 0;
  }
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
  covariance_matrix (2, 0) = covariance_matrix (0, 2);
  covariance_matrix (2, 1) = covariance_matrix (1, 2);

  return (point_count);
}

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
                                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                   Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  unsigned point_count = pcl::computeCovarianceMatrix (cloud, centroid, covariance_matrix);
  if (point_count != 0)
    covariance_matrix /= static_cast<Scalar> (point_count);
  return (point_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixCentroidStandard (const pcl::PointCloud<PointT> &cloud,
                                         const Indices &indices,
                                         const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  covariance_matrix.setZero ();
  // For each point in the cloud
  for (const auto& idx: indices)
  {
    Eigen::Matrix<Scalar, 4, 1> pt;
    pt[0] = cloud[idx].x - centroid[0];
    pt[1] = cloud[idx].y - centroid[1];
    pt[2] = cloud[idx].z - centroid[2];

    covariance_matrix (1, 1) += pt.y () * pt.y ();
    covariance_matrix (1, 2) += pt.y () * pt.z ();

    covariance_matrix (2, 2) += pt.z () * pt.z ();

    pt *= pt.x ();
    covariance_matrix (0, 0) += pt.x ();
    covariance_matrix (0, 1) += pt.y ();
    covariance_matrix (0, 2) += pt.z ();
  }
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
  covariance_matrix (2, 0) = covariance_matrix (0, 2);
  covariance_matrix (2, 1) = covariance_matrix (1, 2);
  return static_cast<unsigned int> (indices.size ());
}

#if defined(__RVV10__)
/** Dense \a cloud with \a indices; \a n < 16 delegates to Standard. */
template <typename PointT, typename Scalar>
inline unsigned int
computeCovarianceMatrixCentroidRVV (const pcl::PointCloud<PointT> &cloud,
                                    const Indices &indices,
                                    const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                    Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  const std::size_t n = indices.size ();
  if (n < 16)
    return computeCovarianceMatrixCentroidStandard (cloud, indices, centroid, covariance_matrix);

  const float cx = static_cast<float> (centroid[0]);
  const float cy = static_cast<float> (centroid[1]);
  const float cz = static_cast<float> (centroid[2]);

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc0 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc1 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc2 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc3 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc4 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc5 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto *base = reinterpret_cast<const std::uint8_t *> (cloud.data ());
  const auto *idx_i32 = reinterpret_cast<const std::int32_t *> (indices.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base, v_off, vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    // FMA + _tu: vd += vs1*vs2; tail undisturbed when vl < VLMAX (see doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const float s0 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc0, v_zero, vlmax));
  const float s1 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc1, v_zero, vlmax));
  const float s2 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc2, v_zero, vlmax));
  const float s3 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc3, v_zero, vlmax));
  const float s4 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc4, v_zero, vlmax));
  const float s5 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc5, v_zero, vlmax));

  covariance_matrix (0, 0) = static_cast<Scalar> (s0);
  covariance_matrix (0, 1) = static_cast<Scalar> (s1);
  covariance_matrix (0, 2) = static_cast<Scalar> (s2);
  covariance_matrix (1, 1) = static_cast<Scalar> (s3);
  covariance_matrix (1, 2) = static_cast<Scalar> (s4);
  covariance_matrix (2, 2) = static_cast<Scalar> (s5);
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
  covariance_matrix (2, 0) = covariance_matrix (0, 2);
  covariance_matrix (2, 1) = covariance_matrix (1, 2);
  return static_cast<unsigned int> (n);
}
#endif // __RVV10__

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                         const Indices &indices,
                         const Eigen::Matrix<Scalar, 4, 1> &centroid,
                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  if (indices.empty ())
    return (0);

  std::size_t point_count;
  // If the data is dense, we don't need to check for NaN
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return computeCovarianceMatrixCentroidRVV (cloud, indices, centroid, covariance_matrix);
#endif
    return computeCovarianceMatrixCentroidStandard (cloud, indices, centroid, covariance_matrix);
  }
  // NaN or Inf values could exist => check for them
  else
  {
    Eigen::Matrix<Scalar, 3, 3> temp_covariance_matrix;
    temp_covariance_matrix.setZero ();
    point_count = 0;
    // For each point in the cloud
    for (const auto& index: indices)
    {
      // Check if the point is invalid
      if (!isFinite (cloud[index]))
        continue;

      Eigen::Matrix<Scalar, 4, 1> pt;
      pt[0] = cloud[index].x - centroid[0];
      pt[1] = cloud[index].y - centroid[1];
      pt[2] = cloud[index].z - centroid[2];

      temp_covariance_matrix (1, 1) += pt.y () * pt.y ();
      temp_covariance_matrix (1, 2) += pt.y () * pt.z ();

      temp_covariance_matrix (2, 2) += pt.z () * pt.z ();

      pt *= pt.x ();
      temp_covariance_matrix (0, 0) += pt.x ();
      temp_covariance_matrix (0, 1) += pt.y ();
      temp_covariance_matrix (0, 2) += pt.z ();
      ++point_count;
    }
    if (point_count > 0) {
      covariance_matrix = temp_covariance_matrix;
    }
  }
  if (point_count == 0) {
    return 0;
  }
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
  covariance_matrix (2, 0) = covariance_matrix (0, 2);
  covariance_matrix (2, 1) = covariance_matrix (1, 2);
  return (static_cast<unsigned int> (point_count));
}

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                         const pcl::PointIndices &indices,
                         const Eigen::Matrix<Scalar, 4, 1> &centroid,
                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  return (pcl::computeCovarianceMatrix (cloud, indices.indices, centroid, covariance_matrix));
}


template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
                                   const Indices &indices,
                                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                   Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  unsigned point_count = pcl::computeCovarianceMatrix (cloud, indices, centroid, covariance_matrix);
  if (point_count != 0)
    covariance_matrix /= static_cast<Scalar> (point_count);

  return (point_count);
}

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
                                   const pcl::PointIndices &indices,
                                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
                                   Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  return computeCovarianceMatrixNormalized(cloud, indices.indices, centroid, covariance_matrix);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// computeCovarianceMatrix (about origin, second moments / n) — dense scalar helpers (OriginStandard) & optional RVV (OriginRVV)
template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixOriginStandard (const pcl::PointCloud<PointT> &cloud,
                                       Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor>::Zero ();

  const unsigned int point_count = static_cast<unsigned int> (cloud.size ());
  // For each point in the cloud
  for (const auto& point: cloud)
  {
    accu[0] += point.x * point.x;
    accu[1] += point.x * point.y;
    accu[2] += point.x * point.z;
    accu[3] += point.y * point.y;
    accu[4] += point.y * point.z;
    accu[5] += point.z * point.z;
  }

  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    covariance_matrix.coeffRef (0) = accu[0];
    covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu[1];
    covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = accu[2];
    covariance_matrix.coeffRef (4) = accu[3];
    covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = accu[4];
    covariance_matrix.coeffRef (8) = accu[5];
  }
  return (point_count);
}

#if defined(__RVV10__)
/** Dense full cloud; sums xx..zz then divides by \a n. \a n < 16 delegates to Standard. */
template <typename PointT, typename Scalar>
inline unsigned int
computeCovarianceMatrixOriginRVV (const pcl::PointCloud<PointT> &cloud,
                                  Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  const std::size_t n = cloud.size ();
  if (n < 16)
    return computeCovarianceMatrixOriginStandard (cloud, covariance_matrix);

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc0 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc1 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc2 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc3 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc4 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc5 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto *base = reinterpret_cast<const std::uint8_t *> (cloud.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    // FMA + _tu: vd += vs1*vs2; tail undisturbed when vl < VLMAX (see doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const float s0 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc0, v_zero, vlmax));
  const float s1 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc1, v_zero, vlmax));
  const float s2 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc2, v_zero, vlmax));
  const float s3 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc3, v_zero, vlmax));
  const float s4 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc4, v_zero, vlmax));
  const float s5 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc5, v_zero, vlmax));

  const Scalar inv_n = static_cast<Scalar> (1) / static_cast<Scalar> (n);
  Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor> accu;
  accu[0] = static_cast<Scalar> (s0) * inv_n;
  accu[1] = static_cast<Scalar> (s1) * inv_n;
  accu[2] = static_cast<Scalar> (s2) * inv_n;
  accu[3] = static_cast<Scalar> (s3) * inv_n;
  accu[4] = static_cast<Scalar> (s4) * inv_n;
  accu[5] = static_cast<Scalar> (s5) * inv_n;
  covariance_matrix.coeffRef (0) = accu[0];
  covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu[1];
  covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = accu[2];
  covariance_matrix.coeffRef (4) = accu[3];
  covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = accu[4];
  covariance_matrix.coeffRef (8) = accu[5];
  return static_cast<unsigned int> (n);
}
#endif // __RVV10__

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor>::Zero ();

  unsigned int point_count;
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return computeCovarianceMatrixOriginRVV (cloud, covariance_matrix);
#endif
    return computeCovarianceMatrixOriginStandard (cloud, covariance_matrix);
  }
  else
  {
    point_count = 0;
    for (const auto& point: cloud)
    {
      if (!isFinite (point))
        continue;

      accu [0] += point.x * point.x;
      accu [1] += point.x * point.y;
      accu [2] += point.x * point.z;
      accu [3] += point.y * point.y;
      accu [4] += point.y * point.z;
      accu [5] += point.z * point.z;
      ++point_count;
    }
  }

  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    covariance_matrix.coeffRef (0) = accu [0];
    covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu [1];
    covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = accu [2];
    covariance_matrix.coeffRef (4) = accu [3];
    covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = accu [4];
    covariance_matrix.coeffRef (8) = accu [5];
  }
  return (point_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrixOriginStandard (const pcl::PointCloud<PointT> &cloud,
                                       const Indices &indices,
                                       Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor>::Zero ();

  const unsigned int point_count = static_cast<unsigned int> (indices.size ());
  for (const auto& index: indices)
  {
    //const PointT& point = cloud[*iIt];
    accu[0] += cloud[index].x * cloud[index].x;
    accu[1] += cloud[index].x * cloud[index].y;
    accu[2] += cloud[index].x * cloud[index].z;
    accu[3] += cloud[index].y * cloud[index].y;
    accu[4] += cloud[index].y * cloud[index].z;
    accu[5] += cloud[index].z * cloud[index].z;
  }
  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    covariance_matrix.coeffRef (0) = accu[0];
    covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu[1];
    covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = accu[2];
    covariance_matrix.coeffRef (4) = accu[3];
    covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = accu[4];
    covariance_matrix.coeffRef (8) = accu[5];
  }
  return (point_count);
}

#if defined(__RVV10__)
/** Dense \a cloud with \a indices; \a n < 16 delegates to Standard. */
template <typename PointT, typename Scalar>
inline unsigned int
computeCovarianceMatrixOriginRVV (const pcl::PointCloud<PointT> &cloud,
                                  const Indices &indices,
                                  Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  const std::size_t n = indices.size ();
  if (n < 16)
    return computeCovarianceMatrixOriginStandard (cloud, indices, covariance_matrix);

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc0 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc1 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc2 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc3 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc4 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc5 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto *base = reinterpret_cast<const std::uint8_t *> (cloud.data ());
  const auto *idx_i32 = reinterpret_cast<const std::int32_t *> (indices.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base, v_off, vl, vx, vy, vz);
    // FMA + _tu: vd += vs1*vs2; tail undisturbed when vl < VLMAX (see doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const float s0 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc0, v_zero, vlmax));
  const float s1 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc1, v_zero, vlmax));
  const float s2 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc2, v_zero, vlmax));
  const float s3 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc3, v_zero, vlmax));
  const float s4 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc4, v_zero, vlmax));
  const float s5 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc5, v_zero, vlmax));

  const Scalar inv_n = static_cast<Scalar> (1) / static_cast<Scalar> (n);
  Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor> accu;
  accu[0] = static_cast<Scalar> (s0) * inv_n;
  accu[1] = static_cast<Scalar> (s1) * inv_n;
  accu[2] = static_cast<Scalar> (s2) * inv_n;
  accu[3] = static_cast<Scalar> (s3) * inv_n;
  accu[4] = static_cast<Scalar> (s4) * inv_n;
  accu[5] = static_cast<Scalar> (s5) * inv_n;
  covariance_matrix.coeffRef (0) = accu[0];
  covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu[1];
  covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = accu[2];
  covariance_matrix.coeffRef (4) = accu[3];
  covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = accu[4];
  covariance_matrix.coeffRef (8) = accu[5];
  return static_cast<unsigned int> (n);
}
#endif // __RVV10__

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                         const Indices &indices,
                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 6, Eigen::RowMajor>::Zero ();

  unsigned int point_count;
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return computeCovarianceMatrixOriginRVV (cloud, indices, covariance_matrix);
#endif
    return computeCovarianceMatrixOriginStandard (cloud, indices, covariance_matrix);
  }
  else
  {
    point_count = 0;
    for (const auto& index: indices)
    {
      if (!isFinite (cloud[index]))
        continue;

      ++point_count;
      accu [0] += cloud[index].x * cloud[index].x;
      accu [1] += cloud[index].x * cloud[index].y;
      accu [2] += cloud[index].x * cloud[index].z;
      accu [3] += cloud[index].y * cloud[index].y;
      accu [4] += cloud[index].y * cloud[index].z;
      accu [5] += cloud[index].z * cloud[index].z;
    }
  }
  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    covariance_matrix.coeffRef (0) = accu [0];
    covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu [1];
    covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = accu [2];
    covariance_matrix.coeffRef (4) = accu [3];
    covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = accu [4];
    covariance_matrix.coeffRef (8) = accu [5];
  }
  return (point_count);
}

template <typename PointT, typename Scalar> inline unsigned int
computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                         const pcl::PointIndices &indices,
                         Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  return (computeCovarianceMatrix (cloud, indices.indices, covariance_matrix));
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrixStandard (const pcl::PointCloud<PointT> &cloud,
                                        Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                        Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // Shifted data/with estimate of mean. This gives very good accuracy and good performance.
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  // Dense path only: split from legacy for RVV; numerics match upstream dense branch.
  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  Eigen::Matrix<Scalar, 3, 1> K (0.0, 0.0, 0.0);
  for (const auto &point : cloud)
    if (isFinite (point))
    {
      K.x () = point.x;
      K.y () = point.y;
      K.z () = point.z;
      break;
    }
  const std::size_t point_count = cloud.size ();
  // For each point in the cloud
  for (const auto &point : cloud)
  {
    const Scalar x = point.x - K.x (), y = point.y - K.y (), z = point.z - K.z ();
    accu[0] += x * x;
    accu[1] += x * y;
    accu[2] += x * z;
    accu[3] += y * y;
    accu[4] += y * z;
    accu[5] += z * z;
    accu[6] += x;
    accu[7] += y;
    accu[8] += z;
  }
  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    centroid[0] = accu[6] + K.x ();
    centroid[1] = accu[7] + K.y ();
    centroid[2] = accu[8] + K.z ();//effective mean E[P=(x,y,z)]
    centroid[3] = 1;
    covariance_matrix.coeffRef (0) = accu[0] - accu[6] * accu[6];//(0,0)xx : E[(x-E[x])^2]=E[x^2]-E[x]^2=E[(x-Kx)^2]-E[x-Kx]^2
    covariance_matrix.coeffRef (1) = accu[1] - accu[6] * accu[7];//(0,1)xy : E[(x-E[x])(y-E[y])]=E[xy]-E[x]E[y]=E[(x-Kx)(y-Ky)]-E[x-Kx]E[y-Ky]
    covariance_matrix.coeffRef (2) = accu[2] - accu[6] * accu[8];//(0,2)xz
    covariance_matrix.coeffRef (4) = accu[3] - accu[7] * accu[7];//(1,1)yy
    covariance_matrix.coeffRef (5) = accu[4] - accu[7] * accu[8];//(1,2)yz
    covariance_matrix.coeffRef (8) = accu[5] - accu[8] * accu[8];//(2,2)zz
    covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);   //(1,0)yx
    covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);   //(2,0)zx
    covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);   //(2,1)zy
  }
  return (static_cast<unsigned int> (point_count));
}

#if defined(__RVV10__)
/** Dense full cloud; \a n < 16 delegates to Standard (same pattern as \c compute3DCentroidRVV). */
template <typename PointT, typename Scalar>
inline unsigned int
computeMeanAndCovarianceMatrixRVV (const pcl::PointCloud<PointT> &cloud,
                                   Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                   Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  const std::size_t n = cloud.size ();
  if (n < 16)
    return computeMeanAndCovarianceMatrixStandard (cloud, covariance_matrix, centroid);

  Eigen::Matrix<Scalar, 3, 1> K (0.0, 0.0, 0.0);
  for (const auto &point : cloud)
    if (isFinite (point))
    {
      K.x () = point.x;
      K.y () = point.y;
      K.z () = point.z;
      break;
    }
  const float kx = static_cast<float> (K.x ());
  const float ky = static_cast<float> (K.y ());
  const float kz = static_cast<float> (K.z ());

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc0 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc1 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc2 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc3 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc4 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc5 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc6 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc7 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc8 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto *base = reinterpret_cast<const std::uint8_t *> (cloud.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, kx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, ky, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, kz, vl);
    // FMA + _tu: vd += vs1*vs2; tail undisturbed when vl < VLMAX (see doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
    v_acc6 = __riscv_vfmacc_vf_f32m2_tu (v_acc6, 1.0f, vx, vl);
    v_acc7 = __riscv_vfmacc_vf_f32m2_tu (v_acc7, 1.0f, vy, vl);
    v_acc8 = __riscv_vfmacc_vf_f32m2_tu (v_acc8, 1.0f, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const float a0 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc0, v_zero, vlmax));
  const float a1 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc1, v_zero, vlmax));
  const float a2 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc2, v_zero, vlmax));
  const float a3 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc3, v_zero, vlmax));
  const float a4 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc4, v_zero, vlmax));
  const float a5 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc5, v_zero, vlmax));
  const float a6 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc6, v_zero, vlmax));
  const float a7 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc7, v_zero, vlmax));
  const float a8 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc8, v_zero, vlmax));

  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu;
  accu[0] = static_cast<Scalar> (a0);
  accu[1] = static_cast<Scalar> (a1);
  accu[2] = static_cast<Scalar> (a2);
  accu[3] = static_cast<Scalar> (a3);
  accu[4] = static_cast<Scalar> (a4);
  accu[5] = static_cast<Scalar> (a5);
  accu[6] = static_cast<Scalar> (a6);
  accu[7] = static_cast<Scalar> (a7);
  accu[8] = static_cast<Scalar> (a8);

  accu /= static_cast<Scalar> (n);
  centroid[0] = accu[6] + K.x ();
  centroid[1] = accu[7] + K.y ();
  centroid[2] = accu[8] + K.z ();//effective mean E[P=(x,y,z)]
  centroid[3] = 1;
  covariance_matrix.coeffRef (0) = accu[0] - accu[6] * accu[6];//(0,0)xx : E[(x-E[x])^2]=E[x^2]-E[x]^2=E[(x-Kx)^2]-E[x-Kx]^2
  covariance_matrix.coeffRef (1) = accu[1] - accu[6] * accu[7];//(0,1)xy : E[(x-E[x])(y-E[y])]=E[xy]-E[x]E[y]=E[(x-Kx)(y-Ky)]-E[x-Kx]E[y-Ky]
  covariance_matrix.coeffRef (2) = accu[2] - accu[6] * accu[8];//(0,2)xz
  covariance_matrix.coeffRef (4) = accu[3] - accu[7] * accu[7];//(1,1)yy
  covariance_matrix.coeffRef (5) = accu[4] - accu[7] * accu[8];//(1,2)yz
  covariance_matrix.coeffRef (8) = accu[5] - accu[8] * accu[8];//(2,2)zz
  covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);   //(1,0)yx
  covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);   //(2,0)zx
  covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);   //(2,1)zy
  return (static_cast<unsigned int> (n));
}
#endif // __RVV10__

template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                                Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return computeMeanAndCovarianceMatrixRVV (cloud, covariance_matrix, centroid);
#endif
    return computeMeanAndCovarianceMatrixStandard (cloud, covariance_matrix, centroid);
  }

  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  Eigen::Matrix<Scalar, 3, 1> K (0.0, 0.0, 0.0);
  for (const auto &point : cloud)
    if (isFinite (point))
    {
      K.x () = point.x;
      K.y () = point.y;
      K.z () = point.z;
      break;
    }
  std::size_t point_count = 0;
  for (const auto &point : cloud)
  {
    if (!isFinite (point))
      continue;

    const Scalar x = point.x - K.x (), y = point.y - K.y (), z = point.z - K.z ();
    accu[0] += x * x;
    accu[1] += x * y;
    accu[2] += x * z;
    accu[3] += y * y;
    accu[4] += y * z;
    accu[5] += z * z;
    accu[6] += x;
    accu[7] += y;
    accu[8] += z;
    ++point_count;
  }
  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    centroid[0] = accu[6] + K.x ();
    centroid[1] = accu[7] + K.y ();
    centroid[2] = accu[8] + K.z ();//effective mean E[P=(x,y,z)]
    centroid[3] = 1;
    covariance_matrix.coeffRef (0) = accu[0] - accu[6] * accu[6];//(0,0)xx : E[(x-E[x])^2]=E[x^2]-E[x]^2=E[(x-Kx)^2]-E[x-Kx]^2
    covariance_matrix.coeffRef (1) = accu[1] - accu[6] * accu[7];//(0,1)xy : E[(x-E[x])(y-E[y])]=E[xy]-E[x]E[y]=E[(x-Kx)(y-Ky)]-E[x-Kx]E[y-Ky]
    covariance_matrix.coeffRef (2) = accu[2] - accu[6] * accu[8];//(0,2)xz
    covariance_matrix.coeffRef (4) = accu[3] - accu[7] * accu[7];//(1,1)yy
    covariance_matrix.coeffRef (5) = accu[4] - accu[7] * accu[8];//(1,2)yz
    covariance_matrix.coeffRef (8) = accu[5] - accu[8] * accu[8];//(2,2)zz
    covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);   //(1,0)yx
    covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);   //(2,0)zx
    covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);   //(2,1)zy
  }
  return (static_cast<unsigned int> (point_count));
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrixStandard (const pcl::PointCloud<PointT> &cloud,
                                        const Indices &indices,
                                        Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                        Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // Shifted data/with estimate of mean. This gives very good accuracy and good performance.
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  // Dense path + indices: split from legacy for RVV.
  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  Eigen::Matrix<Scalar, 3, 1> K (0.0, 0.0, 0.0);
  for (const auto &index : indices)
    if (isFinite (cloud[index]))
    {
      K.x () = cloud[index].x;
      K.y () = cloud[index].y;
      K.z () = cloud[index].z;
      break;
    }
  const std::size_t point_count = indices.size ();
  // For each point in the cloud
  for (const auto &index : indices)
  {
    const Scalar x = cloud[index].x - K.x (), y = cloud[index].y - K.y (), z = cloud[index].z - K.z ();
    accu[0] += x * x;
    accu[1] += x * y;
    accu[2] += x * z;
    accu[3] += y * y;
    accu[4] += y * z;
    accu[5] += z * z;
    accu[6] += x;
    accu[7] += y;
    accu[8] += z;
  }
  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    centroid[0] = accu[6] + K.x ();
    centroid[1] = accu[7] + K.y ();
    centroid[2] = accu[8] + K.z ();//effective mean E[P=(x,y,z)]
    centroid[3] = 1;
    covariance_matrix.coeffRef (0) = accu[0] - accu[6] * accu[6];//(0,0)xx : E[(x-E[x])^2]=E[x^2]-E[x]^2=E[(x-Kx)^2]-E[x-Kx]^2
    covariance_matrix.coeffRef (1) = accu[1] - accu[6] * accu[7];//(0,1)xy : E[(x-E[x])(y-E[y])]=E[xy]-E[x]E[y]=E[(x-Kx)(y-Ky)]-E[x-Kx]E[y-Ky]
    covariance_matrix.coeffRef (2) = accu[2] - accu[6] * accu[8];//(0,2)xz
    covariance_matrix.coeffRef (4) = accu[3] - accu[7] * accu[7];//(1,1)yy
    covariance_matrix.coeffRef (5) = accu[4] - accu[7] * accu[8];//(1,2)yz
    covariance_matrix.coeffRef (8) = accu[5] - accu[8] * accu[8];//(2,2)zz
    covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);   //(1,0)yx
    covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);   //(2,0)zx
    covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);   //(2,1)zy
  }
  return (static_cast<unsigned int> (point_count));
}

#if defined(__RVV10__)
/** Dense \a cloud with \a indices; \a n < 16 delegates to Standard. */
template <typename PointT, typename Scalar>
inline unsigned int
computeMeanAndCovarianceMatrixRVV (const pcl::PointCloud<PointT> &cloud,
                                   const Indices &indices,
                                   Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                   Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  const std::size_t n = indices.size ();
  if (n < 16)
    return computeMeanAndCovarianceMatrixStandard (cloud, indices, covariance_matrix, centroid);

  Eigen::Matrix<Scalar, 3, 1> K (0.0, 0.0, 0.0);
  for (const auto &index : indices)
    if (isFinite (cloud[index]))
    {
      K.x () = cloud[index].x;
      K.y () = cloud[index].y;
      K.z () = cloud[index].z;
      break;
    }
  const float kx = static_cast<float> (K.x ());
  const float ky = static_cast<float> (K.y ());
  const float kz = static_cast<float> (K.z ());

  const std::size_t vlmax = __riscv_vsetvlmax_e32m2 ();
  vfloat32m2_t v_acc0 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc1 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc2 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc3 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc4 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc5 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc6 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc7 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  vfloat32m2_t v_acc8 = __riscv_vfmv_v_f_f32m2 (0.0f, vlmax);
  const auto *base = reinterpret_cast<const std::uint8_t *> (cloud.data ());
  const auto *idx_i32 = reinterpret_cast<const std::int32_t *> (indices.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base, v_off, vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, kx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, ky, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, kz, vl);
    // same FMA / tail policy as full-cloud RVV kernel above
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
    v_acc6 = __riscv_vfmacc_vf_f32m2_tu (v_acc6, 1.0f, vx, vl);
    v_acc7 = __riscv_vfmacc_vf_f32m2_tu (v_acc7, 1.0f, vy, vl);
    v_acc8 = __riscv_vfmacc_vf_f32m2_tu (v_acc8, 1.0f, vz, vl);
    i += vl;
  }

  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const float a0 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc0, v_zero, vlmax));
  const float a1 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc1, v_zero, vlmax));
  const float a2 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc2, v_zero, vlmax));
  const float a3 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc3, v_zero, vlmax));
  const float a4 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc4, v_zero, vlmax));
  const float a5 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc5, v_zero, vlmax));
  const float a6 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc6, v_zero, vlmax));
  const float a7 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc7, v_zero, vlmax));
  const float a8 = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredosum_vs_f32m2_f32m1 (v_acc8, v_zero, vlmax));

  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu;
  accu[0] = static_cast<Scalar> (a0);
  accu[1] = static_cast<Scalar> (a1);
  accu[2] = static_cast<Scalar> (a2);
  accu[3] = static_cast<Scalar> (a3);
  accu[4] = static_cast<Scalar> (a4);
  accu[5] = static_cast<Scalar> (a5);
  accu[6] = static_cast<Scalar> (a6);
  accu[7] = static_cast<Scalar> (a7);
  accu[8] = static_cast<Scalar> (a8);

  accu /= static_cast<Scalar> (n);
  centroid[0] = accu[6] + K.x ();
  centroid[1] = accu[7] + K.y ();
  centroid[2] = accu[8] + K.z ();//effective mean E[P=(x,y,z)]
  centroid[3] = 1;
  covariance_matrix.coeffRef (0) = accu[0] - accu[6] * accu[6];//(0,0)xx : E[(x-E[x])^2]=E[x^2]-E[x]^2=E[(x-Kx)^2]-E[x-Kx]^2
  covariance_matrix.coeffRef (1) = accu[1] - accu[6] * accu[7];//(0,1)xy : E[(x-E[x])(y-E[y])]=E[xy]-E[x]E[y]=E[(x-Kx)(y-Ky)]-E[x-Kx]E[y-Ky]
  covariance_matrix.coeffRef (2) = accu[2] - accu[6] * accu[8];//(0,2)xz
  covariance_matrix.coeffRef (4) = accu[3] - accu[7] * accu[7];//(1,1)yy
  covariance_matrix.coeffRef (5) = accu[4] - accu[7] * accu[8];//(1,2)yz
  covariance_matrix.coeffRef (8) = accu[5] - accu[8] * accu[8];//(2,2)zz
  covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);   //(1,0)yx
  covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);   //(2,0)zx
  covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);   //(2,1)zy
  return (static_cast<unsigned int> (n));
}
#endif // __RVV10__

template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                                const Indices &indices,
                                Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return computeMeanAndCovarianceMatrixRVV (cloud, indices, covariance_matrix, centroid);
#endif
    return computeMeanAndCovarianceMatrixStandard (cloud, indices, covariance_matrix, centroid);
  }

  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  Eigen::Matrix<Scalar, 3, 1> K (0.0, 0.0, 0.0);
  for (const auto &index : indices)
    if (isFinite (cloud[index]))
    {
      K.x () = cloud[index].x;
      K.y () = cloud[index].y;
      K.z () = cloud[index].z;
      break;
    }
  std::size_t point_count = 0;
  for (const auto &index : indices)
  {
    if (!isFinite (cloud[index]))
      continue;

    ++point_count;
    const Scalar x = cloud[index].x - K.x (), y = cloud[index].y - K.y (), z = cloud[index].z - K.z ();
    accu[0] += x * x;
    accu[1] += x * y;
    accu[2] += x * z;
    accu[3] += y * y;
    accu[4] += y * z;
    accu[5] += z * z;
    accu[6] += x;
    accu[7] += y;
    accu[8] += z;
  }

  if (point_count != 0)
  {
    accu /= static_cast<Scalar> (point_count);
    centroid[0] = accu[6] + K.x ();
    centroid[1] = accu[7] + K.y ();
    centroid[2] = accu[8] + K.z ();//effective mean E[P=(x,y,z)]
    centroid[3] = 1;
    covariance_matrix.coeffRef (0) = accu[0] - accu[6] * accu[6];//(0,0)xx : E[(x-E[x])^2]=E[x^2]-E[x]^2=E[(x-Kx)^2]-E[x-Kx]^2
    covariance_matrix.coeffRef (1) = accu[1] - accu[6] * accu[7];//(0,1)xy : E[(x-E[x])(y-E[y])]=E[xy]-E[x]E[y]=E[(x-Kx)(y-Ky)]-E[x-Kx]E[y-Ky]
    covariance_matrix.coeffRef (2) = accu[2] - accu[6] * accu[8];//(0,2)xz
    covariance_matrix.coeffRef (4) = accu[3] - accu[7] * accu[7];//(1,1)yy
    covariance_matrix.coeffRef (5) = accu[4] - accu[7] * accu[8];//(1,2)yz
    covariance_matrix.coeffRef (8) = accu[5] - accu[8] * accu[8];//(2,2)zz
    covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);   //(1,0)yx
    covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);   //(2,0)zx
    covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);   //(2,1)zy
  }
  return (static_cast<unsigned int> (point_count));
}

template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                                const pcl::PointIndices &indices,
                                Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  return (computeMeanAndCovarianceMatrix (cloud, indices.indices, covariance_matrix, centroid));
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
computeCentroidAndOBB (const pcl::PointCloud<PointT> &cloud,
  Eigen::Matrix<Scalar, 3, 1> &centroid,
  Eigen::Matrix<Scalar, 3, 1> &obb_center,
  Eigen::Matrix<Scalar, 3, 1> &obb_dimensions,
  Eigen::Matrix<Scalar, 3, 3> &obb_rotational_matrix)
{
  Eigen::Matrix<Scalar, 3, 3> covariance_matrix;
  Eigen::Matrix<Scalar, 4, 1> centroid4;
  const auto point_count = computeMeanAndCovarianceMatrix(cloud, covariance_matrix, centroid4);
  if (!point_count)
    return (0);
  centroid(0) = centroid4(0);
  centroid(1) = centroid4(1);
  centroid(2) = centroid4(2);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> evd(covariance_matrix);
  const Eigen::Matrix<Scalar, 3, 3> eigenvectors_ = evd.eigenvectors();
  const Eigen::Matrix<Scalar, 3, 1> minor_axis = eigenvectors_.col(0);//the eigenvectors do not need to be normalized (they are already)
  const Eigen::Matrix<Scalar, 3, 1> middle_axis = eigenvectors_.col(1);
  // Enforce right hand rule:
  const Eigen::Matrix<Scalar, 3, 1> major_axis = middle_axis.cross(minor_axis);

  obb_rotational_matrix <<
    major_axis(0), middle_axis(0), minor_axis(0),
    major_axis(1), middle_axis(1), minor_axis(1),
    major_axis(2), middle_axis(2), minor_axis(2);
  //obb_rotational_matrix.col(0)==major_axis
  //obb_rotational_matrix.col(1)==middle_axis
  //obb_rotational_matrix.col(2)==minor_axis

  //Transforming the point cloud in the (Centroid, ma-mi-mi_axis) reference
  //with homogeneous matrix
  //[R^t  , -R^t*Centroid ]
  //[0    , 1             ]
  Eigen::Matrix<Scalar, 4, 4> transform = Eigen::Matrix<Scalar, 4, 4>::Identity();
  transform.template topLeftCorner<3, 3>() = obb_rotational_matrix.transpose();
  transform.template topRightCorner<3, 1>().noalias() =-transform.template topLeftCorner<3, 3>()*centroid;

  //when Scalar==double on a Windows 10 machine and MSVS:
  //if you substitute the following Scalars with floats you get a 20% worse processing time, if with 2 PointT 55% worse
  Scalar obb_min_pointx, obb_min_pointy, obb_min_pointz;
  Scalar obb_max_pointx, obb_max_pointy, obb_max_pointz;
  obb_min_pointx = obb_min_pointy = obb_min_pointz = std::numeric_limits<Scalar>::max();
  obb_max_pointx = obb_max_pointy = obb_max_pointz = std::numeric_limits<Scalar>::min();

  if (cloud.is_dense)
  {
    const auto& point = cloud[0];
    Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
    Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

    obb_min_pointx = obb_max_pointx = P(0);
    obb_min_pointy = obb_max_pointy = P(1);
    obb_min_pointz = obb_max_pointz = P(2);

    for (size_t i=1; i<cloud.size();++i)
    {
      const auto&  point = cloud[i];
      Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
      Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

      if (P(0) < obb_min_pointx)
        obb_min_pointx = P(0);
      else if (P(0) > obb_max_pointx)
        obb_max_pointx = P(0);
      if (P(1) < obb_min_pointy)
        obb_min_pointy = P(1);
      else if (P(1) > obb_max_pointy)
        obb_max_pointy = P(1);
      if (P(2) < obb_min_pointz)
        obb_min_pointz = P(2);
      else if (P(2) > obb_max_pointz)
        obb_max_pointz = P(2);
    }
  }
  else
  {
    size_t i = 0;
    for (; i < cloud.size(); ++i)
    {
      const auto&  point = cloud[i];
      if (!isFinite(point))
        continue;
      Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
      Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

      obb_min_pointx = obb_max_pointx = P(0);
      obb_min_pointy = obb_max_pointy = P(1);
      obb_min_pointz = obb_max_pointz = P(2);
      ++i;
      break;
    }

    for (; i<cloud.size();++i)
    {
      const auto&  point = cloud[i];
      if (!isFinite(point))
        continue;
      Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
      Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

      if (P(0) < obb_min_pointx)
        obb_min_pointx = P(0);
      else if (P(0) > obb_max_pointx)
        obb_max_pointx = P(0);
      if (P(1) < obb_min_pointy)
        obb_min_pointy = P(1);
      else if (P(1) > obb_max_pointy)
        obb_max_pointy = P(1);
      if (P(2) < obb_min_pointz)
        obb_min_pointz = P(2);
      else if (P(2) > obb_max_pointz)
        obb_max_pointz = P(2);
    }

  }

  const Eigen::Matrix<Scalar, 3, 1>  //shift between point cloud centroid and OBB center (position of the OBB center relative to (p.c.centroid, major_axis, middle_axis, minor_axis))
    shift((obb_max_pointx + obb_min_pointx) / 2.0f,
      (obb_max_pointy + obb_min_pointy) / 2.0f,
      (obb_max_pointz + obb_min_pointz) / 2.0f);

  obb_dimensions(0) = obb_max_pointx - obb_min_pointx;
  obb_dimensions(1) = obb_max_pointy - obb_min_pointy;
  obb_dimensions(2) = obb_max_pointz - obb_min_pointz;

  obb_center.noalias() = centroid + obb_rotational_matrix * shift;//position of the OBB center in the same reference Oxyz of the point cloud

  return (point_count);
}

template <typename PointT, typename Scalar> inline unsigned int
computeCentroidAndOBB (const pcl::PointCloud<PointT> &cloud,
  const Indices &indices,
  Eigen::Matrix<Scalar, 3, 1> &centroid,
  Eigen::Matrix<Scalar, 3, 1> &obb_center,
  Eigen::Matrix<Scalar, 3, 1> &obb_dimensions,
  Eigen::Matrix<Scalar, 3, 3> &obb_rotational_matrix)
{
  Eigen::Matrix<Scalar, 3, 3> covariance_matrix;
  Eigen::Matrix<Scalar, 4, 1> centroid4;
  const auto point_count = computeMeanAndCovarianceMatrix(cloud, indices, covariance_matrix, centroid4);
  if (!point_count)
    return (0);
  centroid(0) = centroid4(0);
  centroid(1) = centroid4(1);
  centroid(2) = centroid4(2);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> evd(covariance_matrix);
  const Eigen::Matrix<Scalar, 3, 3> eigenvectors_ = evd.eigenvectors();
  const Eigen::Matrix<Scalar, 3, 1> minor_axis = eigenvectors_.col(0);//the eigenvectors do not need to be normalized (they are already)
  const Eigen::Matrix<Scalar, 3, 1> middle_axis = eigenvectors_.col(1);
  // Enforce right hand rule:
  const Eigen::Matrix<Scalar, 3, 1> major_axis = middle_axis.cross(minor_axis);

  obb_rotational_matrix <<
    major_axis(0), middle_axis(0), minor_axis(0),
    major_axis(1), middle_axis(1), minor_axis(1),
    major_axis(2), middle_axis(2), minor_axis(2);
  //obb_rotational_matrix.col(0)==major_axis
  //obb_rotational_matrix.col(1)==middle_axis
  //obb_rotational_matrix.col(2)==minor_axis

  //Transforming the point cloud in the (Centroid, ma-mi-mi_axis) reference
  //with homogeneous matrix
  //[R^t  , -R^t*Centroid ]
  //[0    , 1             ]
  Eigen::Matrix<Scalar, 4, 4> transform = Eigen::Matrix<Scalar, 4, 4>::Identity();
  transform.template topLeftCorner<3, 3>() = obb_rotational_matrix.transpose();
  transform.template topRightCorner<3, 1>().noalias() =-transform.template topLeftCorner<3, 3>()*centroid;

  //when Scalar==double on a Windows 10 machine and MSVS:
  //if you substitute the following Scalars with floats you get a 20% worse processing time, if with 2 PointT 55% worse
  Scalar obb_min_pointx, obb_min_pointy, obb_min_pointz;
  Scalar obb_max_pointx, obb_max_pointy, obb_max_pointz;
  obb_min_pointx = obb_min_pointy = obb_min_pointz = std::numeric_limits<Scalar>::max();
  obb_max_pointx = obb_max_pointy = obb_max_pointz = std::numeric_limits<Scalar>::min();

  if (cloud.is_dense)
  {
    const auto&  point = cloud[indices[0]];
    Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
    Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

    obb_min_pointx = obb_max_pointx = P(0);
    obb_min_pointy = obb_max_pointy = P(1);
    obb_min_pointz = obb_max_pointz = P(2);

    for (size_t i=1; i<indices.size();++i)
    {
      const auto &  point = cloud[indices[i]];

      Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
      Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

      if (P(0) < obb_min_pointx)
        obb_min_pointx = P(0);
      else if (P(0) > obb_max_pointx)
        obb_max_pointx = P(0);
      if (P(1) < obb_min_pointy)
        obb_min_pointy = P(1);
      else if (P(1) > obb_max_pointy)
        obb_max_pointy = P(1);
      if (P(2) < obb_min_pointz)
        obb_min_pointz = P(2);
      else if (P(2) > obb_max_pointz)
        obb_max_pointz = P(2);
    }
  }
  else
  {
    size_t i = 0;
    for (; i<indices.size();++i)
    {
      const auto&  point = cloud[indices[i]];
      if (!isFinite(point))
        continue;
      Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
      Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

      obb_min_pointx = obb_max_pointx = P(0);
      obb_min_pointy = obb_max_pointy = P(1);
      obb_min_pointz = obb_max_pointz = P(2);
      ++i;
      break;
    }

    for (; i<indices.size();++i)
    {
      const auto&  point = cloud[indices[i]];
      if (!isFinite(point))
        continue;

      Eigen::Matrix<Scalar, 4, 1> P0(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y) , static_cast<Scalar>(point.z), 1.0);
      Eigen::Matrix<Scalar, 4, 1> P = transform * P0;

      if (P(0) < obb_min_pointx)
        obb_min_pointx = P(0);
      else if (P(0) > obb_max_pointx)
        obb_max_pointx = P(0);
      if (P(1) < obb_min_pointy)
        obb_min_pointy = P(1);
      else if (P(1) > obb_max_pointy)
        obb_max_pointy = P(1);
      if (P(2) < obb_min_pointz)
        obb_min_pointz = P(2);
      else if (P(2) > obb_max_pointz)
        obb_max_pointz = P(2);
    }

  }

  const Eigen::Matrix<Scalar, 3, 1>  //shift between point cloud centroid and OBB center (position of the OBB center relative to (p.c.centroid, major_axis, middle_axis, minor_axis))
    shift((obb_max_pointx + obb_min_pointx) / 2.0f,
      (obb_max_pointy + obb_min_pointy) / 2.0f,
      (obb_max_pointz + obb_min_pointz) / 2.0f);

  obb_dimensions(0) = obb_max_pointx - obb_min_pointx;
  obb_dimensions(1) = obb_max_pointy - obb_min_pointy;
  obb_dimensions(2) = obb_max_pointz - obb_min_pointz;

  obb_center.noalias() = centroid + obb_rotational_matrix * shift;//position of the OBB center in the same reference Oxyz of the point cloud

  return (point_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> void
demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  pcl::PointCloud<PointT> &cloud_out,
                  int npts)
{
  // Calculate the number of points if not given
  if (npts == 0)
  {
    while (cloud_iterator.isValid ())
    {
      ++npts;
      ++cloud_iterator;
    }
    cloud_iterator.reset ();
  }

  int i = 0;
  cloud_out.resize (npts);
  // Subtract the centroid from cloud_in
  while (cloud_iterator.isValid ())
  {
    cloud_out[i].x = cloud_iterator->x - centroid[0];
    cloud_out[i].y = cloud_iterator->y - centroid[1];
    cloud_out[i].z = cloud_iterator->z - centroid[2];
    ++i;
    ++cloud_iterator;
  }
  cloud_out.width = cloud_out.size ();
  cloud_out.height = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> void
demeanPointCloudStd (pcl::PointCloud<PointT> &cloud_out,
                     const Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // Subtract the centroid from cloud_in
  for (auto& point: cloud_out)
  {
    point.x -= static_cast<float> (centroid[0]);
    point.y -= static_cast<float> (centroid[1]);
    point.z -= static_cast<float> (centroid[2]);
  }
}

#if defined(__RVV10__)
template <typename PointT, typename Scalar> void
demeanPointCloudRVV (pcl::PointCloud<PointT> &cloud_out,
                     const Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  const std::size_t n = cloud_out.size ();
  if (n < 16)
  {
    demeanPointCloudStd (cloud_out, centroid);
    return;
  }

  const float cx = static_cast<float> (centroid[0]);
  const float cy = static_cast<float> (centroid[1]);
  const float cz = static_cast<float> (centroid[2]);

  auto* base = reinterpret_cast<std::uint8_t*> (cloud_out.data ());
  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    pcl::rvv_store::strided_store3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    i += vl;
  }
}
#endif

template <typename PointT, typename Scalar> void
demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  pcl::PointCloud<PointT> &cloud_out)
{
  cloud_out = cloud_in;

  if (cloud_out.empty ())
    return;

  // Subtract the centroid from cloud_in
  if (cloud_in.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
    {
      demeanPointCloudRVV (cloud_out, centroid);
      return;
    }
#endif
  }
  demeanPointCloudStd (cloud_out, centroid);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> void
demeanPointCloudStd (const pcl::PointCloud<PointT> &cloud_in,
                     const Indices &indices,
                     const Eigen::Matrix<Scalar, 4, 1> &centroid,
                     pcl::PointCloud<PointT> &cloud_out)
{
  cloud_out.header = cloud_in.header;
  cloud_out.is_dense = cloud_in.is_dense;
  if (indices.size () == cloud_in.size ())
  {
    cloud_out.width    = cloud_in.width;
    cloud_out.height   = cloud_in.height;
  }
  else
  {
    cloud_out.width    = indices.size ();
    cloud_out.height   = 1;
  }
  cloud_out.resize (indices.size ());

  // Subtract the centroid from cloud_in
  for (std::size_t i = 0; i < indices.size (); ++i)
  {
    cloud_out[i].x = static_cast<float> (cloud_in[indices[i]].x - centroid[0]);
    cloud_out[i].y = static_cast<float> (cloud_in[indices[i]].y - centroid[1]);
    cloud_out[i].z = static_cast<float> (cloud_in[indices[i]].z - centroid[2]);
  }
}

#if defined(__RVV10__)
template <typename PointT, typename Scalar> void
demeanPointCloudRVV (const pcl::PointCloud<PointT> &cloud_in,
                     const Indices &indices,
                     const Eigen::Matrix<Scalar, 4, 1> &centroid,
                     pcl::PointCloud<PointT> &cloud_out)
{
  const std::size_t n = indices.size ();
  if (n < 16)
  {
    demeanPointCloudStd (cloud_in, indices, centroid, cloud_out);
    return;
  }

  cloud_out.header = cloud_in.header;
  cloud_out.is_dense = cloud_in.is_dense;
  if (indices.size () == cloud_in.size ())
  {
    cloud_out.width    = cloud_in.width;
    cloud_out.height   = cloud_in.height;
  }
  else
  {
    cloud_out.width    = indices.size ();
    cloud_out.height   = 1;
  }
  cloud_out.resize (indices.size ());

  const float cx = static_cast<float> (centroid[0]);
  const float cy = static_cast<float> (centroid[1]);
  const float cz = static_cast<float> (centroid[2]);

  const auto* in_base = reinterpret_cast<const std::uint8_t*> (cloud_in.data ());
  auto* out_base = reinterpret_cast<std::uint8_t*> (cloud_out.data ());
  const auto* idx_i32 = reinterpret_cast<const std::int32_t*> (indices.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off_in = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);

    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        in_base, v_off_in, vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    pcl::rvv_store::strided_store3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        out_base + i * sizeof (PointT), vl, vx, vy, vz);
    i += vl;
  }
}
#endif

template <typename PointT, typename Scalar> void
demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                  const Indices &indices,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  pcl::PointCloud<PointT> &cloud_out)
{
  if (cloud_in.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
    {
      demeanPointCloudRVV (cloud_in, indices, centroid, cloud_out);
      return;
    }
#endif
  }
  demeanPointCloudStd (cloud_in, indices, centroid, cloud_out);
}


template <typename PointT, typename Scalar> void
demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                  const pcl::PointIndices& indices,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  pcl::PointCloud<PointT> &cloud_out)
{
  return (demeanPointCloud (cloud_in, indices.indices, centroid, cloud_out));
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> void
demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out,
                  int npts)
{
  // Calculate the number of points if not given
  if (npts == 0)
  {
    while (cloud_iterator.isValid ())
    {
      ++npts;
      ++cloud_iterator;
    }
    cloud_iterator.reset ();
  }

  cloud_out = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>::Zero (4, npts);        // keep the data aligned

  int i = 0;
  while (cloud_iterator.isValid ())
  {
    cloud_out (0, i) = cloud_iterator->x - centroid[0];
    cloud_out (1, i) = cloud_iterator->y - centroid[1];
    cloud_out (2, i) = cloud_iterator->z - centroid[2];
    ++i;
    ++cloud_iterator;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> void
demeanPointCloudEigenStd (const pcl::PointCloud<PointT> &cloud_in,
                          const Eigen::Matrix<Scalar, 4, 1> &centroid,
                          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  const std::size_t npts = cloud_in.size ();
  cloud_out = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>::Zero (4, static_cast<Eigen::Index> (npts));        // keep the data aligned

  for (std::size_t i = 0; i < npts; ++i)
  {
    cloud_out (0, i) = cloud_in[i].x - centroid[0];
    cloud_out (1, i) = cloud_in[i].y - centroid[1];
    cloud_out (2, i) = cloud_in[i].z - centroid[2];
    // One column at a time
    //cloud_out.block<4, 1> (0, i) = cloud_in[i].getVector4fMap () - centroid;
  }

  // Make sure we zero the 4th dimension out (1 row, N columns)
  //cloud_out.block (3, 0, 1, npts).setZero ();
}

#if defined(__RVV10__)
template <typename PointT>
void
demeanPointCloudEigenRVV (const pcl::PointCloud<PointT> &cloud_in,
                          const Eigen::Matrix<float, 4, 1> &centroid,
                          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  const std::size_t n = cloud_in.size ();
  if (n < 16)
  {
    demeanPointCloudEigenStd (cloud_in, centroid, cloud_out);
    return;
  }

  const float cx = centroid[0];
  const float cy = centroid[1];
  const float cz = centroid[2];

  cloud_out = Eigen::Matrix<float, 4, Eigen::Dynamic>::Zero (4, static_cast<Eigen::Index> (n));

  float* row0 = &cloud_out (0, 0);
  float* row1 = &cloud_out (1, 0);
  float* row2 = &cloud_out (2, 0);

  constexpr std::size_t kColStrideBytes = 4 * sizeof (float);

  const auto* in_base = reinterpret_cast<const std::uint8_t*> (cloud_in.data ());

  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        in_base + i * sizeof (PointT), vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    pcl::rvv_store::strided_store_f32m2<kColStrideBytes> (row0 + i * 4, vx, vl);
    pcl::rvv_store::strided_store_f32m2<kColStrideBytes> (row1 + i * 4, vy, vl);
    pcl::rvv_store::strided_store_f32m2<kColStrideBytes> (row2 + i * 4, vz, vl);
    i += vl;
  }
}
#endif

template <typename PointT, typename Scalar> void
demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  const std::size_t npts = cloud_in.size ();

  if (cloud_in.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
    {
      if constexpr (std::is_same_v<Scalar, float>)
      {
        demeanPointCloudEigenRVV (cloud_in, centroid, cloud_out);
        return;
      }
    }
#endif
  }
  demeanPointCloudEigenStd (cloud_in, centroid, cloud_out);
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> void
demeanPointCloudEigenStd (const pcl::PointCloud<PointT> &cloud_in,
                          const Indices &indices,
                          const Eigen::Matrix<Scalar, 4, 1> &centroid,
                          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  const std::size_t npts = indices.size ();
  cloud_out = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>::Zero (4, static_cast<Eigen::Index> (npts));        // keep the data aligned

  for (std::size_t i = 0; i < npts; ++i)
  {
    cloud_out (0, i) = cloud_in[indices[i]].x - centroid[0];
    cloud_out (1, i) = cloud_in[indices[i]].y - centroid[1];
    cloud_out (2, i) = cloud_in[indices[i]].z - centroid[2];
    // One column at a time
    //cloud_out.block<4, 1> (0, i) = cloud_in[indices[i]].getVector4fMap () - centroid;
  }

  // Make sure we zero the 4th dimension out (1 row, N columns)
  //cloud_out.block (3, 0, 1, npts).setZero ();
}

#if defined(__RVV10__)
template <typename PointT>
void
demeanPointCloudEigenRVV (const pcl::PointCloud<PointT> &cloud_in,
                          const Indices &indices,
                          const Eigen::Matrix<float, 4, 1> &centroid,
                          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  const std::size_t n = indices.size ();
  if (n < 16)
  {
    demeanPointCloudEigenStd (cloud_in, indices, centroid, cloud_out);
    return;
  }

  const float cx = centroid[0];
  const float cy = centroid[1];
  const float cz = centroid[2];

  cloud_out = Eigen::Matrix<float, 4, Eigen::Dynamic>::Zero (4, static_cast<Eigen::Index> (n));

  float* row0 = &cloud_out (0, 0);
  float* row1 = &cloud_out (1, 0);
  float* row2 = &cloud_out (2, 0);

  constexpr std::size_t kColStrideBytes = 4 * sizeof (float);

  const auto* in_base = reinterpret_cast<const std::uint8_t*> (cloud_in.data ());
  const auto* idx_i32 = reinterpret_cast<const std::int32_t*> (indices.data ());

  std::size_t col = 0;
  while (col < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - col);
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + col, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off_in = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);

    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        in_base, v_off_in, vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    pcl::rvv_store::strided_store_f32m2<kColStrideBytes> (row0 + col * 4, vx, vl);
    pcl::rvv_store::strided_store_f32m2<kColStrideBytes> (row1 + col * 4, vy, vl);
    pcl::rvv_store::strided_store_f32m2<kColStrideBytes> (row2 + col * 4, vz, vl);
    col += vl;
  }
}
#endif

template <typename PointT, typename Scalar> void
demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                  const Indices &indices,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  if (cloud_in.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
    {
      if constexpr (std::is_same_v<Scalar, float>)
      {
        demeanPointCloudEigenRVV (cloud_in, indices, centroid, cloud_out);
        return;
      }
    }
#endif
  }
  demeanPointCloudEigenStd (cloud_in, indices, centroid, cloud_out);
}

template <typename PointT, typename Scalar> void
demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                  const pcl::PointIndices &indices,
                  const Eigen::Matrix<Scalar, 4, 1> &centroid,
                  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out)
{
  return (pcl::demeanPointCloud (cloud_in, indices.indices, centroid, cloud_out));
}

///////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline void
computeNDCentroid (const pcl::PointCloud<PointT> &cloud,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid)
{
  using FieldList = typename pcl::traits::fieldList<PointT>::type;

  // Get the size of the fields
  centroid.setZero (boost::mpl::size<FieldList>::value);

  if (cloud.empty ())
    return;

  // Iterate over each point
  for (const auto& pt: cloud)
  {
    // Iterate over each dimension
    pcl::for_each_type<FieldList> (NdCentroidFunctor<PointT, Scalar> (pt, centroid));
  }
  centroid /= static_cast<Scalar> (cloud.size ());
}


template <typename PointT, typename Scalar> inline void
computeNDCentroid (const pcl::PointCloud<PointT> &cloud,
                   const Indices &indices,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid)
{
  using FieldList = typename pcl::traits::fieldList<PointT>::type;

  // Get the size of the fields
  centroid.setZero (boost::mpl::size<FieldList>::value);

  if (indices.empty ())
    return;

  // Iterate over each point
  for (const auto& index: indices)
  {
    // Iterate over each dimension
    pcl::for_each_type<FieldList> (NdCentroidFunctor<PointT, Scalar> (cloud[index], centroid));
  }
  centroid /= static_cast<Scalar> (indices.size ());
}


template <typename PointT, typename Scalar> inline void
computeNDCentroid (const pcl::PointCloud<PointT> &cloud,
                   const pcl::PointIndices &indices,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid)
{
  return (pcl::computeNDCentroid (cloud, indices.indices, centroid));
}

template <typename PointT> void
CentroidPoint<PointT>::add (const PointT& point)
{
  // Invoke add point on each accumulator
  boost::fusion::for_each (accumulators_, detail::AddPoint<PointT> (point));
  ++num_points_;
}

template <typename PointT>
template <typename PointOutT> void
CentroidPoint<PointT>::get (PointOutT& point) const
{
  if (num_points_ != 0)
  {
    // Filter accumulators so that only those that are compatible with
    // both PointT and requested point type remain
    auto ca = boost::fusion::filter_if<detail::IsAccumulatorCompatible<PointT, PointOutT>> (accumulators_);
    // Invoke get point on each accumulator in filtered list
    boost::fusion::for_each (ca, detail::GetPoint<PointOutT> (point, num_points_));
  }
}


template <typename PointInT, typename PointOutT> std::size_t
computeCentroid (const pcl::PointCloud<PointInT>& cloud,
                      PointOutT& centroid)
{
  pcl::CentroidPoint<PointInT> cp;

  if (cloud.is_dense)
    for (const auto& point: cloud)
      cp.add (point);
  else
    for (const auto& point: cloud)
      if (pcl::isFinite (point))
        cp.add (point);

  cp.get (centroid);
  return (cp.getSize ());
}


template <typename PointInT, typename PointOutT> std::size_t
computeCentroid (const pcl::PointCloud<PointInT>& cloud,
                      const Indices& indices,
                      PointOutT& centroid)
{
  pcl::CentroidPoint<PointInT> cp;

  if (cloud.is_dense)
    for (const auto &index : indices)
      cp.add (cloud[index]);
  else
    for (const auto &index : indices)
      if (pcl::isFinite (cloud[index]))
        cp.add (cloud[index]);

  cp.get (centroid);
  return (cp.getSize ());
}

} // namespace pcl

