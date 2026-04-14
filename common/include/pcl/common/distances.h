/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
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

#include <limits>

#include <pcl/types.h>
#include <pcl/point_types.h> // for PointXY
#include <Eigen/Core> // for VectorXf
#include <Eigen/Geometry> // for MatrixBase::cross3() definition (used in sqrPointToLineDistance)

#if defined(__RVV10__)
#include <riscv_vector.h>
#include <cstddef>
#include <vector>

#include <pcl/common/rvv_point_load.h>
#endif

/**
  * \file pcl/common/distances.h
  * Define standard C methods to do distance calculations
  * \ingroup common
  */

/*@{*/
namespace pcl
{
  template <typename PointT> class PointCloud;

  /** \brief Get the shortest 3D segment between two 3D lines
    * \param line_a the coefficients of the first line (point, direction)
    * \param line_b the coefficients of the second line (point, direction)
    * \param pt1_seg the first point on the line segment
    * \param pt2_seg the second point on the line segment
    * \ingroup common
    */
  PCL_EXPORTS void
  lineToLineSegment (const Eigen::VectorXf &line_a, const Eigen::VectorXf &line_b,
                     Eigen::Vector4f &pt1_seg, Eigen::Vector4f &pt2_seg);

  /** \brief Get the square distance from a point to a line (represented by a point and a direction)
    * \param pt a point
    * \param line_pt a point on the line (make sure that line_pt[3] = 0 as there are no internal checks!)
    * \param line_dir the line direction
    * \ingroup common
    */
  double inline
  sqrPointToLineDistance (const Eigen::Vector4f &pt, const Eigen::Vector4f &line_pt, const Eigen::Vector4f &line_dir)
  {
    // Calculate the distance from the point to the line
    // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p1-p0)) / norm(p2-p1)
    return (line_dir.cross3 (line_pt - pt)).squaredNorm () / line_dir.squaredNorm ();
  }

  /** \brief Get the square distance from a point to a line (represented by a point and a direction)
    * \note This one is useful if one has to compute many distances to a fixed line, so the vector length can be pre-computed
    * \param pt a point
    * \param line_pt a point on the line (make sure that line_pt[3] = 0 as there are no internal checks!)
    * \param line_dir the line direction
    * \param sqr_length the squared norm of the line direction
    * \ingroup common
    */
  double inline
  sqrPointToLineDistance (const Eigen::Vector4f &pt, const Eigen::Vector4f &line_pt, const Eigen::Vector4f &line_dir, const double sqr_length)
  {
    // Calculate the distance from the point to the line
    // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p1-p0)) / norm(p2-p1)
    return (line_dir.cross3 (line_pt - pt)).squaredNorm () / sqr_length;
  }

  /** \brief Obtain the maximum segment in a given set of points, and return the minimum and maximum points.
    * \param[in] cloud the point cloud dataset
    * \param[out] pmin the coordinates of the "minimum" point in \a cloud (one end of the segment)
    * \param[out] pmax the coordinates of the "maximum" point in \a cloud (the other end of the segment)
    * \return the length of segment length
    * \ingroup common
    */
  template <typename PointT> double inline
  getMaxSegment (const pcl::PointCloud<PointT> &cloud,
                  PointT &pmin, PointT &pmax)
  {
#if defined(__RVV10__)
  return getMaxSegmentRVV (cloud, pmin, pmax);
#else
  return getMaxSegmentStandard (cloud, pmin, pmax);
#endif
  }

#if defined(__RVV10__)
  template <typename PointT> double inline
  getMaxSegmentRVV (const pcl::PointCloud<PointT> &cloud,
                    PointT &pmin, PointT &pmax)
  {
    const std::size_t n = cloud.size ();
    if (n < 512)
      return getMaxSegmentStandard (cloud, pmin, pmax);

    float max_dist = -std::numeric_limits<float>::infinity ();
    const auto token = std::numeric_limits<std::size_t>::max();
    std::size_t i_min = token, i_max = token;

    for (std::size_t i = 0; i < n; ++i)
    {
      const float xi = cloud[i].x;
      const float yi = cloud[i].y;
      const float zi = cloud[i].z;

      std::size_t j = i;
      while (j < n)
      {
        const std::size_t vl = __riscv_vsetvl_e32m2 (n - j);

        vfloat32m2_t vx, vy, vz;
        pcl::rvv_load::strided_load3_f32m2<sizeof (PointT),
                                            offsetof (PointT, x),
                                            offsetof (PointT, y),
                                            offsetof (PointT, z)> (
            reinterpret_cast<const std::uint8_t*>(&cloud[j]), vl, vx, vy, vz);

        vx = __riscv_vfsub_vf_f32m2 (vx, xi, vl);
        vy = __riscv_vfsub_vf_f32m2 (vy, yi, vl);
        vz = __riscv_vfsub_vf_f32m2 (vz, zi, vl);

        vfloat32m2_t vdist2 = __riscv_vfmul_vv_f32m2 (vx, vx, vl);
        vdist2 = __riscv_vfmacc_vv_f32m2 (vdist2, vy, vy, vl);
        vdist2 = __riscv_vfmacc_vv_f32m2 (vdist2, vz, vz, vl);

        const vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1 (max_dist, 1);
        const vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m2_f32m1 (vdist2, vinit, vl);
        const float vmax = __riscv_vfmv_f_s_f32m1_f32 (vmax1);

        if (vmax > max_dist)
        {
          const vbool16_t m = __riscv_vmfeq_vf_f32m2_b16 (vdist2, vmax, vl);
          const long lane = __riscv_vfirst_m_b16 (m, vl);
          if (lane >= 0)
          {
            max_dist = vmax;
            i_min = i;
            i_max = j + static_cast<std::size_t>(lane);
          }
        }

        j += vl;
      }
    }

    if (i_min == token || i_max == token)
      return std::numeric_limits<double>::min ();

    pmin = cloud[i_min];
    pmax = cloud[i_max];
    return std::sqrt (static_cast<double>(max_dist));
  }
#endif

  template <typename PointT> double inline
  getMaxSegmentStandard (const pcl::PointCloud<PointT> &cloud,
                 PointT &pmin, PointT &pmax)
  {
    double max_dist = std::numeric_limits<double>::min ();
    const auto token = std::numeric_limits<std::size_t>::max();
    std::size_t i_min = token, i_max = token;

    for (std::size_t i = 0; i < cloud.size (); ++i)
    {
      for (std::size_t j = i; j < cloud.size (); ++j)
      {
        // Compute the distance
        double dist = (cloud[i].getVector4fMap () -
                       cloud[j].getVector4fMap ()).squaredNorm ();
        if (dist <= max_dist)
          continue;

        max_dist = dist;
        i_min = i;
        i_max = j;
      }
    }

    if (i_min == token || i_max == token)
      return std::numeric_limits<double>::min ();

    pmin = cloud[i_min];
    pmax = cloud[i_max];
    return (std::sqrt (max_dist));
  }

  /** \brief Obtain the maximum segment in a given set of points, and return the minimum and maximum points.
    * \param[in] cloud the point cloud dataset
    * \param[in] indices a set of point indices to use from \a cloud
    * \param[out] pmin the coordinates of the "minimum" point in \a cloud (one end of the segment)
    * \param[out] pmax the coordinates of the "maximum" point in \a cloud (the other end of the segment)
    * \return the length of segment length
    * \ingroup common
    */
  template <typename PointT> double inline
  getMaxSegment (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
                 PointT &pmin, PointT &pmax)
  {
#if defined(__RVV10__)
    return getMaxSegmentRVV (cloud, indices, pmin, pmax);
#else
    return getMaxSegmentStandard (cloud, indices, pmin, pmax);
#endif
  }

  template <typename PointT> double inline
  getMaxSegmentStandard (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
                         PointT &pmin, PointT &pmax)
  {
    double max_dist = std::numeric_limits<double>::min ();
    const auto token = std::numeric_limits<std::size_t>::max();
    std::size_t i_min = token, i_max = token;

    for (std::size_t i = 0; i < indices.size (); ++i)
    {
      for (std::size_t j = i; j < indices.size (); ++j)
      {
        // Compute the distance
        double dist = (cloud[indices[i]].getVector4fMap () -
                       cloud[indices[j]].getVector4fMap ()).squaredNorm ();
        if (dist <= max_dist)
          continue;

        max_dist = dist;
        i_min = i;
        i_max = j;
      }
    }

    if (i_min == token || i_max == token)
      return std::numeric_limits<double>::min ();

    pmin = cloud[indices[i_min]];
    pmax = cloud[indices[i_max]];
    return std::sqrt (max_dist);
  }

#if defined(__RVV10__)
  template <typename PointT> double inline
  getMaxSegmentRVV (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
                    PointT &pmin, PointT &pmax)
  {
    const std::size_t n = indices.size ();
    if (n == 0)
      return std::numeric_limits<double>::min ();

    if (n < 512)
      return getMaxSegmentStandard (cloud, indices, pmin, pmax);

    // 将 indices 对应点打包为连续内存，避免在 O(n^2) 循环中做 gather load。
    // 注意：pcl::PointCloud<PointT>::points 使用 Eigen::aligned_allocator；
    // 这里直接 resize + 赋值，避免与 std::vector<PointT> 的 allocator 不匹配。
    pcl::PointCloud<PointT> packed_cloud;
    packed_cloud.resize (n);
    packed_cloud.width = static_cast<std::uint32_t>(n);
    packed_cloud.height = 1;
    packed_cloud.is_dense = true;
    for (std::size_t k = 0; k < n; ++k)
      packed_cloud[k] = cloud[indices[k]];

    return getMaxSegmentRVV (packed_cloud, pmin, pmax);
  }
#endif

  /** \brief Calculate the squared euclidean distance between the two given points.
    * \param[in] p1 the first point
    * \param[in] p2 the second point
    */
  template<typename PointType1, typename PointType2> inline float
  squaredEuclideanDistance (const PointType1& p1, const PointType2& p2)
  {
    float diff_x = p2.x - p1.x, diff_y = p2.y - p1.y, diff_z = p2.z - p1.z;
    return (diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
  }

  /** \brief Calculate the squared euclidean distance between the two given points.
    * \param[in] p1 the first point
    * \param[in] p2 the second point
    */
  template<> inline float
  squaredEuclideanDistance (const PointXY& p1, const PointXY& p2)
  {
    float diff_x = p2.x - p1.x, diff_y = p2.y - p1.y;
    return (diff_x*diff_x + diff_y*diff_y);
  }

   /** \brief Calculate the euclidean distance between the two given points.
    * \param[in] p1 the first point
    * \param[in] p2 the second point
    */
  template<typename PointType1, typename PointType2> inline float
  euclideanDistance (const PointType1& p1, const PointType2& p2)
  {
    return (std::sqrt (squaredEuclideanDistance (p1, p2)));
  }
}
