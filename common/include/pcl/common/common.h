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

#ifdef __SSE__
#include <xmmintrin.h> // for __m128
#endif // ifdef __SSE__
#ifdef __AVX__
#include <immintrin.h> // for __m256
#endif // ifdef __AVX__
#ifdef __RVV10__
#include <riscv_vector.h> // for RISC-V Vector intrinsics
#endif // ifdef __RVV10__

#include <pcl/point_cloud.h> // for PointCloud
#include <pcl/PointIndices.h> // for PointIndices
namespace pcl { struct PCLPointCloud2; }

/**
  * \file pcl/common/common.h
  * Define standard C methods and C++ classes that are common to all methods
  * \ingroup common
  */

/*@{*/
namespace pcl
{
  /** \brief Compute the smallest angle between two 3D vectors in radians (default) or degree.
    * \param v1 the first 3D vector (represented as a \a Eigen::Vector4f)
    * \param v2 the second 3D vector (represented as a \a Eigen::Vector4f)
    * \param in_degree determine if angle should be in radians or degrees
    * \return the angle between v1 and v2 in radians or degrees
    * \note Handles rounding error for parallel and anti-parallel vectors
    * \ingroup common
    */
  inline double
  getAngle3D (const Eigen::Vector4f &v1, const Eigen::Vector4f &v2, const bool in_degree = false);

  /** \brief Compute the smallest angle between two 3D vectors in radians (default) or degree.
    * \param v1 the first 3D vector (represented as a \a Eigen::Vector3f)
    * \param v2 the second 3D vector (represented as a \a Eigen::Vector3f)
    * \param in_degree determine if angle should be in radians or degrees
    * \return the angle between v1 and v2 in radians or degrees
    * \ingroup common
    */
  inline double
  getAngle3D (const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const bool in_degree = false);

#ifdef __SSE__
  /** \brief Compute the approximate arccosine of four values at once using SSE instructions.
    *
    * The approximation used is \f$ (1.59121552+x*(-0.15461442+x*0.05354897))*\sqrt{0.89286965-0.89282669*x}+0.06681017+x*(-0.09402311+x*0.02708663) \f$.
    * The average error is 0.00012 rad. This approximation is more accurate than other approximations of acos, but also uses a few more operations.
    * \param x four floats, each should be in [0; 1]. They must not be greater than 1 since acos is undefined there.
    *          They should not be less than 0 because there the approximation is less precise
    * \return the four arccosines, each in [0; pi/2]
    * \ingroup common
    */
  inline __m128
  acos_SSE (const __m128 &x);

  /** \brief Similar to getAngle3D, but four times in parallel using SSE instructions.
    *
    * This behaves like \f$ min(getAngle3D(dot_product), \pi-getAngle3D(dot_product)) \f$.
    * All vectors must be normalized (length is 1.0).
    * Since an approximate acos is used, the results may be slightly imprecise.
    * \param[in] the x components of the first four vectors
    * \param[in] the y components of the first four vectors
    * \param[in] the z components of the first four vectors
    * \param[in] the x components of the second four vectors
    * \param[in] the y components of the second four vectors
    * \param[in] the z components of the second four vectors
    * \return the four angles in radians in [0; pi/2]
    * \ingroup common
    */
  inline __m128
  getAcuteAngle3DSSE (const __m128 &x1, const __m128 &y1, const __m128 &z1, const __m128 &x2, const __m128 &y2, const __m128 &z2);
#endif // ifdef __SSE__

#ifdef __AVX__
  /** \brief Compute the approximate arccosine of eight values at once using AVX instructions.
    *
    * The approximation used is \f$ (1.59121552+x*(-0.15461442+x*0.05354897))*\sqrt{0.89286965-0.89282669*x}+0.06681017+x*(-0.09402311+x*0.02708663) \f$.
    * The average error is 0.00012 rad. This approximation is more accurate than other approximations of acos, but also uses a few more operations.
    * \param x eight floats, each should be in [0; 1]. They must not be greater than 1 since acos is undefined there.
    *          They should not be less than 0 because there the approximation is less precise
    * \return the eight arccosines, each in [0; pi/2]
    * \ingroup common
    */
  inline __m256
  acos_AVX (const __m256 &x);

  /** \brief Similar to getAngle3D, but eight times in parallel using AVX instructions.
    *
    * This behaves like \f$ min(getAngle3D(dot_product), \pi-getAngle3D(dot_product)) \f$.
    * All vectors must be normalized (length is 1.0).
    * Since an approximate acos is used, the results may be slightly imprecise.
    * \param[in] the x components of the first eight vectors
    * \param[in] the y components of the first eight vectors
    * \param[in] the z components of the first eight vectors
    * \param[in] the x components of the second eight vectors
    * \param[in] the y components of the second eight vectors
    * \param[in] the z components of the second eight vectors
    * \return the eight angles in radians in [0; pi/2]
    * \ingroup common
    */
  inline __m256
  getAcuteAngle3DAVX (const __m256 &x1, const __m256 &y1, const __m256 &z1, const __m256 &x2, const __m256 &y2, const __m256 &z2);
#endif // ifdef __AVX__

#ifdef __RVV10__
  /**
   * \brief Compute the approximate arccosine of multiple values at once using RISC-V
   * Vector instructions.
   *
   * The approximation used is:
   * \f$
   * (1.59121552+x*(-0.15461442+x*0.05354897))*\sqrt{0.89286965-0.89282669*x}+0.06681017+x*(-0.09402311+x*0.02708663)
   * \f$ The average error is ~0.00012 rad.
   * \param x input vector of floats in [0; 1]
   * \param vl vector length
   * \return vector of acos(x) in [0; pi/2]
   * \ingroup common
   */
  inline vfloat32m2_t
  acos_RVV_f32m2(const vfloat32m2_t& x, const std::size_t vl);

  /**
   * \brief Compute acute angles between two sets of normalized 3D vectors using RVV.
   *
   * This behaves like min(angle, π - angle), returning values in [0; π/2].
   * All input vectors must be normalized.
   * \param[in] x1, y1, z1 components of first set of unit vectors
   * \param[in] x2, y2, z2 components of second set of unit vectors
   * \param vl vector length
   * \return vector of acute angles in radians
   * \ingroup common
   */
  inline vfloat32m2_t
  getAcuteAngle3DRVV_f32m2(const vfloat32m2_t& x1, const vfloat32m2_t& y1, const vfloat32m2_t& z1,
                     const vfloat32m2_t& x2, const vfloat32m2_t& y2, const vfloat32m2_t& z2, const std::size_t vl);

  /**
   * \brief Compute atan2(y, x) for multiple values at once using RISC-V Vector instructions.
   *
   * Polynomial approximation (Hastings-style, ~0.01 deg max error). Returns angle in [-pi, pi].
   * \param y numerator (y coordinates)
   * \param x denominator (x coordinates)
   * \param vl vector length
   * \return vector of atan2(y, x) in radians
   * \ingroup common
   *
   * Implementation reference: https://mazzo.li/posts/vectorized-atan2.html
   */
  inline vfloat32m2_t
  atan2_RVV_f32m2(const vfloat32m2_t& y, const vfloat32m2_t& x, const std::size_t vl);

  /**
   * \brief Compute expf(x) for multiple float values using RISC-V Vector (RVV) instructions.
   *
   * Uses "reduction → approximation → reconstruction": reduce x = n*ln2 + r with r in
   * [-ln2/2, ln2/2], approximate exp(r) by a Remez polynomial, then exp(x) = 2^n * exp(r).
   * Max relative error vs std::expf is about 1.5e-6. Inputs are clamped to [-88, 88].
   * \param x vector of float inputs
   * \param vl vector length (from vsetvl)
   * \return vector of exp(x)
   * \ingroup common
   * \see doc-rvv/expf_RVV.zh.md
   */
  inline vfloat32m2_t
  expf_RVV_f32m2(const vfloat32m2_t& x, const std::size_t vl);

  /**
   * \brief Compute exp(x) for multiple values at once using RISC-V Vector instructions.
   *
   * NOTE: exp_RVV_f64m1 (double) has been removed due to accuracy issues.
   * See test-rvv/2d/exp.cpp for the saved implementation. Use expf_RVV_f32m2 for float.
   */
  // inline vfloat64m1_t
  // exp_RVV_f64m1(const vfloat64m1_t& x, const std::size_t vl);
#endif // ifdef __RVV10__

  /** \brief Compute both the mean and the standard deviation of an array of values
    * \param values the array of values
    * \param mean the resultant mean of the distribution
    * \param stddev the resultant standard deviation of the distribution
    * \ingroup common
    */
  inline void
  getMeanStd (const std::vector<float> &values, double &mean, double &stddev);

  /** \brief Get a set of points residing in a box given its bounds
    * \param cloud the point cloud data message
    * \param min_pt the minimum bounds
    * \param max_pt the maximum bounds
    * \param indices the resultant set of point indices residing in the box
    * \ingroup common
    */
  template <typename PointT> inline void
  getPointsInBox (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt,
                  Eigen::Vector4f &max_pt, Indices &indices);

  /** \brief Get the point at maximum distance from a given point and a given pointcloud
    * \param cloud the point cloud data message
    * \param pivot_pt the point from where to compute the distance
    * \param max_pt the point in cloud that is the farthest point away from pivot_pt
    * \ingroup common
    */
  template<typename PointT> inline void
  getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the point at maximum distance from a given point and a given pointcloud
    * \param cloud the point cloud data message
    * \param indices the vector of point indices to use from \a cloud
    * \param pivot_pt the point from where to compute the distance
    * \param max_pt the point in cloud that is the farthest point away from pivot_pt
    * \ingroup common
    */
  template<typename PointT> inline void
  getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
                  const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud. This is the axis aligned bounding box (AABB).
    * \param[in] cloud the point cloud data message
    * \param[out] min_pt the resultant minimum bounds
    * \param[out] max_pt the resultant maximum bounds
    * \ingroup common
    */
  template <typename PointT> inline void
  getMinMax3D (const pcl::PointCloud<PointT> &cloud, PointT &min_pt, PointT &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud. This is the axis aligned bounding box (AABB).
    * \param[in] cloud the point cloud data message
    * \param[out] min_pt the resultant minimum bounds
    * \param[out] max_pt the resultant maximum bounds
    * \ingroup common
    */
  template <typename PointT> inline void
  getMinMax3D (const pcl::PointCloud<PointT> &cloud,
               Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud. This is the axis aligned bounding box (AABB).
    * \param[in] cloud the point cloud data message
    * \param[in] indices the vector of point indices to use from \a cloud
    * \param[out] min_pt the resultant minimum bounds
    * \param[out] max_pt the resultant maximum bounds
    * \ingroup common
    */
  template <typename PointT> inline void
  getMinMax3D (const pcl::PointCloud<PointT> &cloud, const Indices &indices,
               Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud. This is the axis aligned bounding box (AABB).
    * \param[in] cloud the point cloud data message
    * \param[in] indices the vector of point indices to use from \a cloud
    * \param[out] min_pt the resultant minimum bounds
    * \param[out] max_pt the resultant maximum bounds
    * \ingroup common
    */
  template <typename PointT> inline void
  getMinMax3D (const pcl::PointCloud<PointT> &cloud, const pcl::PointIndices &indices,
               Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

  /** \brief Compute the radius of a circumscribed circle for a triangle formed of three points pa, pb, and pc
    * \param pa the first point
    * \param pb the second point
    * \param pc the third point
    * \return the radius of the circumscribed circle
    * \ingroup common
    */
  template <typename PointT> inline double
  getCircumcircleRadius (const PointT &pa, const PointT &pb, const PointT &pc);

  /** \brief Get the minimum and maximum values on a point histogram
    * \param histogram the point representing a multi-dimensional histogram
    * \param len the length of the histogram
    * \param min_p the resultant minimum
    * \param max_p the resultant maximum
    * \ingroup common
    */
  template <typename PointT> inline void
  getMinMax (const PointT &histogram, int len, float &min_p, float &max_p);

  /** \brief Calculate the area of a polygon given a point cloud that defines the polygon
	  * \param polygon point cloud that contains those vertices that comprises the polygon. Vertices are stored in counterclockwise.
	  * \return the polygon area
	  * \ingroup common
	  */
  template<typename PointT> inline float
  calculatePolygonArea (const pcl::PointCloud<PointT> &polygon);

  /** \brief Get the minimum and maximum values on a point histogram
    * \param cloud the cloud containing multi-dimensional histograms
    * \param idx point index representing the histogram that we need to compute min/max for
    * \param field_name the field name containing the multi-dimensional histogram
    * \param min_p the resultant minimum
    * \param max_p the resultant maximum
    * \ingroup common
    */
  PCL_EXPORTS void
  getMinMax (const pcl::PCLPointCloud2 &cloud, int idx, const std::string &field_name,
             float &min_p, float &max_p);

  /** \brief Compute both the mean and the standard deviation of an array of values
    * \param values the array of values
    * \param mean the resultant mean of the distribution
    * \param stddev the resultant standard deviation of the distribution
    * \ingroup common
    */
  PCL_EXPORTS void
  getMeanStdDev (const std::vector<float> &values, double &mean, double &stddev);

  /** \brief Compute the median of a list of values (fast). If the number of values is even, take the mean of the two middle values.
    * This function can be used like this:
    * \code{.cpp}
    * std::vector<double> vector{1.0, 25.0, 9.0, 4.0, 16.0};
    * const double median = pcl::computeMedian (vector.begin (), vector.end (), static_cast<double(*)(double)>(std::sqrt)); // = 3
    * \endcode
    * \param[in,out] begin,end Iterators that mark the beginning and end of the value range. These values will be reordered!
    * \param[in] f A lamda, function pointer, or similar that is implicitly applied to all values before median computation. In reality, it will be applied lazily (i.e. at most twice) and thus may not change the sorting order (e.g. monotonic functions like sqrt are allowed)
    * \return the median
    * \ingroup common
    */
  template<typename IteratorT, typename Functor> inline auto
  computeMedian (IteratorT begin, IteratorT end, Functor f) noexcept ->
  #if __cpp_lib_is_invocable
  std::invoke_result_t<Functor, decltype(*begin)>
  #else
  std::result_of_t<Functor(decltype(*begin))>
  #endif
  {
    const std::size_t size = std::distance(begin, end);
    const std::size_t mid = size/2;
    if (size%2==0)
    { // Even number of values
      std::nth_element (begin, begin + (mid-1), end);
      return (f(begin[mid-1]) + f(*(std::min_element (begin + mid, end)))) / 2.0;
    }
    else
    { // Odd number of values
      std::nth_element (begin, begin + mid, end);
      return f(begin[mid]);
    }
  }

  /** \brief Compute the median of a list of values (fast). See the other overloaded function for more information.
    */
  template<typename IteratorT> inline auto
  computeMedian (IteratorT begin, IteratorT end) noexcept -> typename std::iterator_traits<IteratorT>::value_type
  {
    return computeMedian (begin, end, [](const auto& x){return x;});
  }
}
/*@}*/
#include <pcl/common/impl/common.hpp>
