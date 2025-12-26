/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2010, Willow Garage, Inc.
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

#ifndef PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_NORMAL_PLANE_H_
#define PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_NORMAL_PLANE_H_

#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/common/common.h> // for getAngle3D

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::selectWithinDistance (
      const Eigen::VectorXf &model_coefficients, const double threshold, Indices &inliers)
{
  if (!normals_)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelNormalPlane::selectWithinDistance] No input dataset containing normals was given!\n");
    inliers.clear ();
    return;
  }

  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
    inliers.clear ();
    return;
  }

  // Obtain the plane normal
  Eigen::Vector4f coeff = model_coefficients;
  coeff[3] = 0.0f;

  inliers.clear ();
  error_sqr_dists_.clear ();
  inliers.reserve (indices_->size ());
  error_sqr_dists_.reserve (indices_->size ());

  // Iterate through the 3d points and calculate the distances from them to the plane
  for (std::size_t i = 0; i < indices_->size (); ++i)
  {
    const PointT  &pt = (*input_)[(*indices_)[i]];
    const PointNT &nt = (*normals_)[(*indices_)[i]];
    // Calculate the distance from the point to the plane normal as the dot product
    // D = (P-A).N/|N|
    Eigen::Vector4f p (pt.x, pt.y, pt.z, 0.0f);
    Eigen::Vector4f n (nt.normal_x, nt.normal_y, nt.normal_z, 0.0f);
    double d_euclid = std::abs (coeff.dot (p) + model_coefficients[3]);

    // Calculate the angular distance between the point normal and the plane normal
    double d_normal = std::abs (getAngle3D (n, coeff));
    d_normal = (std::min) (d_normal, M_PI - d_normal);

    // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
    double weight = normal_distance_weight_ * (1.0 - nt.curvature);

    double distance = std::abs (weight * d_normal + (1.0 - weight) * d_euclid);
    if (distance < threshold)
    {
      // Returns the indices of the points whose distances are smaller than the threshold
      inliers.push_back ((*indices_)[i]);
      error_sqr_dists_.push_back (distance);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::countWithinDistance (
      const Eigen::VectorXf &model_coefficients, const double threshold) const
{
  if (!normals_)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelNormalPlane::countWithinDistance] No input dataset containing normals was given!\n");
    return (0);
  }

  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
    return (0);

#if defined (__AVX__) && defined (__AVX2__)
  return countWithinDistanceAVX (model_coefficients, threshold);
#elif defined (__SSE__) && defined (__SSE2__) && defined (__SSE4_1__)
  return countWithinDistanceSSE (model_coefficients, threshold);
#elif defined (__RVV10__)
  return countWithinDistanceRVV (model_coefficients, threshold);
#else
  return countWithinDistanceStandard (model_coefficients, threshold);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::countWithinDistanceStandard (
      const Eigen::VectorXf &model_coefficients, const double threshold, std::size_t i) const
{
  std::size_t nr_p = 0;

  // Obtain the plane normal
  Eigen::Vector4f coeff = model_coefficients;
  coeff[3] = 0.0f;

  // Iterate through the 3d points and calculate the distances from them to the plane
  for (; i < indices_->size (); ++i)
  {
    const PointT  &pt = (*input_)[(*indices_)[i]];
    const PointNT &nt = (*normals_)[(*indices_)[i]];
    // Calculate the distance from the point to the plane normal as the dot product
    // D = (P-A).N/|N|
    const Eigen::Vector4f p (pt.x, pt.y, pt.z, 0.0f);
    const Eigen::Vector4f n (nt.normal_x, nt.normal_y, nt.normal_z, 0.0f);
    const double d_euclid = std::abs (coeff.dot (p) + model_coefficients[3]);

    // Calculate the angular distance between the point normal and the plane normal
    double d_normal = std::abs (getAngle3D (n, coeff));
    d_normal = (std::min) (d_normal, M_PI - d_normal);

    // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
    const double weight = normal_distance_weight_ * (1.0 - nt.curvature);

    if (std::abs (weight * d_normal + (1.0 - weight) * d_euclid) < threshold)
    {
      nr_p++;
    }
  }
  return (nr_p);
}

//////////////////////////////////////////////////////////////////////////
#if defined (__SSE__) && defined (__SSE2__) && defined (__SSE4_1__)
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::countWithinDistanceSSE (
      const Eigen::VectorXf &model_coefficients, const double threshold, std::size_t i) const
{
  std::size_t nr_p = 0;
  const __m128 a_vec = _mm_set1_ps (model_coefficients[0]);
  const __m128 b_vec = _mm_set1_ps (model_coefficients[1]);
  const __m128 c_vec = _mm_set1_ps (model_coefficients[2]);
  const __m128 d_vec = _mm_set1_ps (model_coefficients[3]);
  const __m128 threshold_vec = _mm_set1_ps (threshold);
  const __m128 normal_distance_weight_vec = _mm_set1_ps (normal_distance_weight_);
  const __m128 abs_help = _mm_set1_ps (-0.0F); // -0.0F (negative zero) means that all bits are 0, only the sign bit is 1
  __m128i res = _mm_set1_epi32(0); // This corresponds to nr_p: 4 32bit integers that, summed together, hold the number of inliers
  for (; (i + 4) <= indices_->size (); i += 4)
  {
    const __m128 d_euclid_vec = pcl::SampleConsensusModelPlane<PointT>::dist4 (i, a_vec, b_vec, c_vec, d_vec, abs_help);

    const __m128 d_normal_vec = getAcuteAngle3DSSE (
                                  _mm_set_ps ((*normals_)[(*indices_)[i  ]].normal_x,
                                              (*normals_)[(*indices_)[i+1]].normal_x,
                                              (*normals_)[(*indices_)[i+2]].normal_x,
                                              (*normals_)[(*indices_)[i+3]].normal_x),
                                  _mm_set_ps ((*normals_)[(*indices_)[i  ]].normal_y,
                                              (*normals_)[(*indices_)[i+1]].normal_y,
                                              (*normals_)[(*indices_)[i+2]].normal_y,
                                              (*normals_)[(*indices_)[i+3]].normal_y),
                                  _mm_set_ps ((*normals_)[(*indices_)[i  ]].normal_z,
                                              (*normals_)[(*indices_)[i+1]].normal_z,
                                              (*normals_)[(*indices_)[i+2]].normal_z,
                                              (*normals_)[(*indices_)[i+3]].normal_z),
                                  a_vec, b_vec, c_vec);
    const __m128 weight_vec = _mm_mul_ps (normal_distance_weight_vec, _mm_sub_ps (_mm_set1_ps (1.0f),
                                  _mm_set_ps ((*normals_)[(*indices_)[i  ]].curvature,
                                              (*normals_)[(*indices_)[i+1]].curvature,
                                              (*normals_)[(*indices_)[i+2]].curvature,
                                              (*normals_)[(*indices_)[i+3]].curvature)));
    const __m128 dist = _mm_andnot_ps (abs_help, _mm_add_ps (_mm_mul_ps (weight_vec, d_normal_vec), _mm_mul_ps (_mm_sub_ps (_mm_set1_ps (1.0f), weight_vec), d_euclid_vec)));
    const __m128 mask = _mm_cmplt_ps (dist, threshold_vec); // The mask contains 1 bits if the corresponding points are inliers, else 0 bits
    res = _mm_add_epi32 (res, _mm_and_si128 (_mm_set1_epi32 (1), _mm_castps_si128 (mask))); // The latter part creates a vector with ones (as 32bit integers) where the points are inliers
  }
  nr_p += _mm_extract_epi32 (res, 0);
  nr_p += _mm_extract_epi32 (res, 1);
  nr_p += _mm_extract_epi32 (res, 2);
  nr_p += _mm_extract_epi32 (res, 3);

  // Process the remaining points (at most 3)
  nr_p += countWithinDistanceStandard(model_coefficients, threshold, i);
  return (nr_p);
}
#endif

//////////////////////////////////////////////////////////////////////////
#if defined (__AVX__) && defined (__AVX2__)
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::countWithinDistanceAVX (
      const Eigen::VectorXf &model_coefficients, const double threshold, std::size_t i) const
{
  std::size_t nr_p = 0;
  const __m256 a_vec = _mm256_set1_ps (model_coefficients[0]);
  const __m256 b_vec = _mm256_set1_ps (model_coefficients[1]);
  const __m256 c_vec = _mm256_set1_ps (model_coefficients[2]);
  const __m256 d_vec = _mm256_set1_ps (model_coefficients[3]);
  const __m256 threshold_vec = _mm256_set1_ps (threshold);
  const __m256 normal_distance_weight_vec = _mm256_set1_ps (normal_distance_weight_);
  const __m256 abs_help = _mm256_set1_ps (-0.0F); // -0.0F (negative zero) means that all bits are 0, only the sign bit is 1
  __m256i res = _mm256_set1_epi32(0); // This corresponds to nr_p: 8 32bit integers that, summed together, hold the number of inliers
  for (; (i + 8) <= indices_->size (); i += 8)
  {
    const __m256 d_euclid_vec = pcl::SampleConsensusModelPlane<PointT>::dist8 (i, a_vec, b_vec, c_vec, d_vec, abs_help);

    const __m256 d_normal_vec = getAcuteAngle3DAVX (
                                  _mm256_set_ps ((*normals_)[(*indices_)[i  ]].normal_x,
                                                 (*normals_)[(*indices_)[i+1]].normal_x,
                                                 (*normals_)[(*indices_)[i+2]].normal_x,
                                                 (*normals_)[(*indices_)[i+3]].normal_x,
                                                 (*normals_)[(*indices_)[i+4]].normal_x,
                                                 (*normals_)[(*indices_)[i+5]].normal_x,
                                                 (*normals_)[(*indices_)[i+6]].normal_x,
                                                 (*normals_)[(*indices_)[i+7]].normal_x),
                                  _mm256_set_ps ((*normals_)[(*indices_)[i  ]].normal_y,
                                                 (*normals_)[(*indices_)[i+1]].normal_y,
                                                 (*normals_)[(*indices_)[i+2]].normal_y,
                                                 (*normals_)[(*indices_)[i+3]].normal_y,
                                                 (*normals_)[(*indices_)[i+4]].normal_y,
                                                 (*normals_)[(*indices_)[i+5]].normal_y,
                                                 (*normals_)[(*indices_)[i+6]].normal_y,
                                                 (*normals_)[(*indices_)[i+7]].normal_y),
                                  _mm256_set_ps ((*normals_)[(*indices_)[i  ]].normal_z,
                                                 (*normals_)[(*indices_)[i+1]].normal_z,
                                                 (*normals_)[(*indices_)[i+2]].normal_z,
                                                 (*normals_)[(*indices_)[i+3]].normal_z,
                                                 (*normals_)[(*indices_)[i+4]].normal_z,
                                                 (*normals_)[(*indices_)[i+5]].normal_z,
                                                 (*normals_)[(*indices_)[i+6]].normal_z,
                                                 (*normals_)[(*indices_)[i+7]].normal_z),
                                  a_vec, b_vec, c_vec);
    const __m256 weight_vec = _mm256_mul_ps (normal_distance_weight_vec, _mm256_sub_ps (_mm256_set1_ps (1.0f),
                                  _mm256_set_ps ((*normals_)[(*indices_)[i  ]].curvature,
                                                 (*normals_)[(*indices_)[i+1]].curvature,
                                                 (*normals_)[(*indices_)[i+2]].curvature,
                                                 (*normals_)[(*indices_)[i+3]].curvature,
                                                 (*normals_)[(*indices_)[i+4]].curvature,
                                                 (*normals_)[(*indices_)[i+5]].curvature,
                                                 (*normals_)[(*indices_)[i+6]].curvature,
                                                 (*normals_)[(*indices_)[i+7]].curvature)));
    const __m256 dist = _mm256_andnot_ps (abs_help, _mm256_add_ps (_mm256_mul_ps (weight_vec, d_normal_vec), _mm256_mul_ps (_mm256_sub_ps (_mm256_set1_ps (1.0f), weight_vec), d_euclid_vec)));
    const __m256 mask = _mm256_cmp_ps (dist, threshold_vec, _CMP_LT_OQ); // The mask contains 1 bits if the corresponding points are inliers, else 0 bits
    res = _mm256_add_epi32 (res, _mm256_and_si256 (_mm256_set1_epi32 (1), _mm256_castps_si256 (mask))); // The latter part creates a vector with ones (as 32bit integers) where the points are inliers
  }
  nr_p += _mm256_extract_epi32 (res, 0);
  nr_p += _mm256_extract_epi32 (res, 1);
  nr_p += _mm256_extract_epi32 (res, 2);
  nr_p += _mm256_extract_epi32 (res, 3);
  nr_p += _mm256_extract_epi32 (res, 4);
  nr_p += _mm256_extract_epi32 (res, 5);
  nr_p += _mm256_extract_epi32 (res, 6);
  nr_p += _mm256_extract_epi32 (res, 7);

  // Process the remaining points (at most 7)
  nr_p += countWithinDistanceStandard(model_coefficients, threshold, i);
  return (nr_p);
}
#endif

#if defined (__RVV10__)
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::countWithinDistanceRVV (
      const Eigen::VectorXf &model_coefficients, const double threshold, std::size_t i) const
{
  std::size_t nr_p = 0;
  const std::size_t total_n = indices_->size();

  // Plane coefficients
  const float a = model_coefficients[0];
  const float b = model_coefficients[1];
  const float c = model_coefficients[2];
  const float d = model_coefficients[3];

  // Threshold and Weights
  // Explicitly cast to float because RVV operations will occur in single precision.
  const float th = static_cast<float>(threshold);
  const float w_scalar = static_cast<float>(normal_distance_weight_);

  // Pointer to the start of the indices vector
  const pcl::index_t* indices_ptr = indices_->data();

  // Pointers to the start of PointT and PointNT data arrays.
  // We use reinterpret_cast<const uint8_t*> to perform precise byte-level
  // pointer arithmetic later (base + offset).
  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(input_->points.data());
  const uint8_t* normals_base = reinterpret_cast<const uint8_t*>(normals_->points.data());

  // Loop through all points using RVV strip-mining.
  for (; i < total_n; ) {

    // Configure Vector Length (Strip-mining)
    // Ask hardware to process as many elements as possible (up to VLMAX)
    // based on the remaining count (total_n - i).
    // e32m2: Element width 32-bit, Register Grouping (LMUL) = 2.
    const size_t vl = __riscv_vsetvl_e32m2(total_n - i);

    // Load Indices Once
    // Load 'vl' indices from the indices vector.
    // These indices will be reused for both Point gathering and Normal gathering.
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2((const uint32_t*)(indices_ptr + i), vl);

    // Broadcast Coefficients
    // Replicate scalar plane coefficients into vector registers.
    const vfloat32m2_t v_a = __riscv_vfmv_v_f_f32m2(a, vl);
    const vfloat32m2_t v_b = __riscv_vfmv_v_f_f32m2(b, vl);
    const vfloat32m2_t v_c = __riscv_vfmv_v_f_f32m2(c, vl);
    const vfloat32m2_t v_d = __riscv_vfmv_v_f_f32m2(d, vl);

    // Calculate byte offsets for PointT: offset = index * sizeof(PointT)
    const vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);

    // Gather X, Y, Z coordinates using Unordered Indexed Load (vluxei32).
    // The base address is adjusted by the struct member offset (offsetof).
    const vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2((const float*)(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2((const float*)(points_base + offsetof(PointT, y)), v_off_pt, vl);
    const vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2((const float*)(points_base + offsetof(PointT, z)), v_off_pt, vl);

    // Calculate Euclidean distance using the math kernel helper.
    // Data is already in registers, avoiding re-fetching.
    const vfloat32m2_t v_d_euc = pcl::SampleConsensusModelPlane<PointT>::distRVV(v_px, v_py, v_pz, v_a, v_b, v_c, v_d, vl);

    // Calculate byte offsets for PointNT.
    // Note: PointNT size might differ from PointT, but we reuse v_idx.
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    // Gather Normal components and Curvature.
    const vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2((const float*)(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2((const float*)(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    const vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2((const float*)(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);
    const vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2((const float*)(normals_base + offsetof(PointNT, curvature)), v_off_norm, vl);

    // Calculate the acute angle between point normal and plane normal.
    // (Assumes getAcuteAngle3DRVV is implemented similarly to the SSE/AVX versions)
    const vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV(v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    // Calculate weight: weight = w_scalar * (1.0 - curvature)
    const vfloat32m2_t v_w = __riscv_vfmul_vf_f32m2(
                          __riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl),
                          w_scalar,
                          vl);

    // Calculate final distance: dist = w * d_norm + (1 - w) * d_euc
    vfloat32m2_t v_dist = __riscv_vfmacc_vv_f32m2(
                            __riscv_vfmul_vv_f32m2(v_w, v_d_norm, vl),
                            __riscv_vfrsub_vf_f32m2(v_w, 1.0f, vl),
                            v_d_euc,
                            vl);
    v_dist = __riscv_vfsgnjx_vv_f32m2(v_dist, v_dist, vl); // Absolute value

    // Create a boolean mask where dist < threshold.
    // vmflt.vf: Vector Mask Float Less-Than Scalar.
    const vbool16_t v_mask = __riscv_vmflt_vf_f32m2_b16(v_dist, th, vl);

    // Count the set bits in the mask (Population Count).
    nr_p += __riscv_vcpop_m_b16(v_mask, vl);

    // Advance the loop index by the number of processed elements (vl).
    i += vl;
  }

  // RVV is Vector Length Agnostic, and there are no remaining points that need to be processed using countWithinDistanceStandard.

  return nr_p;
}
#endif // defined (__RVV10__)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::getDistancesToModel (
      const Eigen::VectorXf &model_coefficients, std::vector<double> &distances) const
{
  if (!normals_)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelNormalPlane::getDistancesToModel] No input dataset containing normals was given!\n");
    return;
  }

  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
    distances.clear ();
    return;
  }

  // Obtain the plane normal
  Eigen::Vector4f coeff = model_coefficients;
  coeff[3] = 0.0f;

  distances.resize (indices_->size ());

  // Iterate through the 3d points and calculate the distances from them to the plane
  for (std::size_t i = 0; i < indices_->size (); ++i)
  {
    const PointT  &pt = (*input_)[(*indices_)[i]];
    const PointNT &nt = (*normals_)[(*indices_)[i]];
    // Calculate the distance from the point to the plane normal as the dot product
    // D = (P-A).N/|N|
    Eigen::Vector4f p (pt.x, pt.y, pt.z, 0.0f);
    Eigen::Vector4f n (nt.normal_x, nt.normal_y, nt.normal_z, 0.0f);
    double d_euclid = std::abs (coeff.dot (p) + model_coefficients[3]);

    // Calculate the angular distance between the point normal and the plane normal
    double d_normal = std::abs (getAngle3D (n, coeff));
    d_normal = (std::min) (d_normal, M_PI - d_normal);

    // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
    double weight = normal_distance_weight_ * (1.0 - nt.curvature);

    distances[i] = std::abs (weight * d_normal + (1.0 - weight) * d_euclid);
  }
}

#define PCL_INSTANTIATE_SampleConsensusModelNormalPlane(PointT, PointNT) template class PCL_EXPORTS pcl::SampleConsensusModelNormalPlane<PointT, PointNT>;

#endif    // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_NORMAL_PLANE_H_

