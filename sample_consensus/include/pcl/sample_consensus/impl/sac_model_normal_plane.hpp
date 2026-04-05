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

  // --- 内存预分配的关键步骤 ---
  inliers.clear();
  error_sqr_dists_.clear();
  // 注意：使用 resize 而不是 reserve，以便我们可以直接通过指针写入
  inliers.resize(indices_->size());
  error_sqr_dists_.resize(indices_->size());

  std::size_t nr_p = 0; // 记录实际内点数量

#if defined (__RVV10__)
  // 调用 RVV 版本，返回找到的内点数量
  nr_p = selectWithinDistanceRVV(model_coefficients, threshold, inliers);
#else
  // 调用标准版本
  nr_p = selectWithinDistanceStandard(model_coefficients, threshold, inliers, 0, 0);
#endif

  // --- 收缩内存 ---
  // 将容器大小调整为实际内点数量
  inliers.resize(nr_p);
  error_sqr_dists_.resize(nr_p);
}

// Standard 实现 (支持 offset，用于处理 RVV 剩下的尾部，或者纯标量运行)
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::selectWithinDistanceStandard (
      const Eigen::VectorXf &model_coefficients, const double threshold,
      Indices &inliers, std::size_t i, std::size_t current_count)
{
  // 预处理系数 coeff
  Eigen::Vector4f coeff = model_coefficients;
  coeff[3] = 0.0f;

  // 遍历每一个点（使用索引数组 indices_）
  for (; i < indices_->size (); ++i)
  {
    // 双重间接寻址：
    // indices_[i] -> 获取点在原始云中的真实 ID (Real ID)
    // (*input_)[Real ID] -> 获取具体的坐标点 PointT
    // (*normals_)[Real ID] -> 获取对应的法线点 PointNT
    const PointT  &pt = (*input_)[(*indices_)[i]];
    const PointNT &nt = (*normals_)[(*indices_)[i]];

    // 计算欧氏距离 (Euclidean Distance)
    // 公式：D_geom = |ax + by + cz + d|
    // coeff.dot(p) 计算 ax + by + cz
    // model_coefficients[3] 是 d
    Eigen::Vector4f p (pt.x, pt.y, pt.z, 0.0f);
    // 注意：这里定义的 n 只是为了给 getAngle3D 传参，实际上 NormalPlane 需要更高效的计算
    Eigen::Vector4f n (nt.normal_x, nt.normal_y, nt.normal_z, 0.0f);
    double d_euclid = std::abs (coeff.dot (p) + model_coefficients[3]);

    // 计算角度差异 (Angular Distance)
    // 计算点法线 n 与平面法线 coeff 的夹角 (弧度)
    // getAngle3D 内部通常计算 acos(dot(n1, n2))
    double d_normal = std::abs (getAngle3D (n, coeff));
    // 处理钝角情况：我们只关心锐角差异。如果夹角 > 90度 (PI/2)，说明法线反向了，取补角
    // 实际上 PCL 这里写的 min(d, PI-d) 是为了把范围限制在 [0, PI/2]
    d_normal = (std::min) (d_normal, M_PI - d_normal);

    // 计算权重
    // 基于曲率 (curvature) 的权重。
    // 如果曲率小（平坦表面），权重高，更多参考法线距离；
    // 如果曲率大（噪点或边缘），权重低，更多参考欧氏距离。
    double weight = normal_distance_weight_ * (1.0 - nt.curvature);

    // 综合距离计算
    // 线性插值混合两种距离
    double distance = std::abs (weight * d_normal + (1.0 - weight) * d_euclid);

    if (distance < threshold)
    {
      // 直接写入预分配的内存
      inliers[current_count] = (*indices_)[i];
      error_sqr_dists_[current_count] = distance;
      current_count++;
    }
  }
  return current_count;
}

#if defined (__RVV10__)
/** \brief RVV implementation of \c selectWithinDistance for the normal-plane model (gather, mask, compress).
 *
 * \note Precision: the weighted distance is computed in \c float (\c vfloat32m2_t) end-to-end; the scalar path
 *       (\ref selectWithinDistanceStandard) uses \c double intermediates with \c getAngle3D. The threshold is
 *       compared after casting to \c float. Euclidean and angular terms use \c distRVV_f32m2 and
 *       \c pcl::getAcuteAngle3DRVV_f32m2 (approximate \c acos); stored distances are widened to \c double.
 *       Results may differ from the scalar implementation near thresholds or for ill-conditioned geometry.
 */
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::selectWithinDistanceRVV (
      const Eigen::VectorXf &model_coefficients, const double threshold, Indices &inliers)
{
  std::size_t i = 0;
  std::size_t nr_p = 0; // 全局内点计数器，也作为写入内存的 Offset
  const std::size_t total_n = indices_->size();

  // 1. 系数提取与广播准备 (与 countWithinDistanceRVV 一致)
  const float a = model_coefficients[0];
  const float b = model_coefficients[1];
  const float c = model_coefficients[2];
  const float d = model_coefficients[3];
  const float th = static_cast<float>(threshold);
  const float w_scalar = static_cast<float>(normal_distance_weight_);

  const pcl::index_t* indices_ptr = indices_->data();
  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(input_->points.data());
  const uint8_t* normals_base = reinterpret_cast<const uint8_t*>(normals_->points.data());

  // 2. 获取输出数组的原始指针 (因为我们已经 resize 过了，直接写内存是安全的)
  int* inliers_out_ptr = inliers.data();
  double* dists_out_ptr = error_sqr_dists_.data();

  // VLA Loop
  for (; i < total_n; ) {
    // 动态设定向量长度
    const size_t vl = __riscv_vsetvl_e32m2(total_n - i);

    // --- A. 加载索引 ---
    const vuint32m2_t v_idx = __riscv_vle32_v_u32m2((const uint32_t*)(indices_ptr + i), vl);
    const vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    // --- B. 加载数据 (Gather Load) ---
    // 加载 PointT (x, y, z)
    const vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, y)), v_off_pt, vl);
    const vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, z)), v_off_pt, vl);

    // 加载 PointNT (nx, ny, nz)
    const vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    const vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);

    // 加载 Curvature (单独加载)
    const vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, curvature)), v_off_norm, vl);

    // --- C. 计算距离 (全程 Float) ---
    // 广播系数
    const vfloat32m2_t v_a = __riscv_vfmv_v_f_f32m2(a, vl);
    const vfloat32m2_t v_b = __riscv_vfmv_v_f_f32m2(b, vl);
    const vfloat32m2_t v_c = __riscv_vfmv_v_f_f32m2(c, vl);
    const vfloat32m2_t v_d = __riscv_vfmv_v_f_f32m2(d, vl);

    // Math Kernels
    const vfloat32m2_t v_d_euc = pcl::SampleConsensusModelPlane<PointT>::distRVV_f32m2(v_px, v_py, v_pz, v_a, v_b, v_c, v_d, vl);
    const vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV_f32m2(v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    // Weight Calculation
    const vfloat32m2_t v_w = __riscv_vfmul_vf_f32m2(
                          __riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl), w_scalar, vl);

    // Final Distance
    vfloat32m2_t v_dist = __riscv_vfmacc_vv_f32m2(
                            __riscv_vfmul_vv_f32m2(v_w, v_d_norm, vl),
                            __riscv_vfrsub_vf_f32m2(v_w, 1.0f, vl), v_d_euc, vl);
    v_dist = __riscv_vfsgnjx_vv_f32m2(v_dist, v_dist, vl); // abs(dist)

    // --- D. 筛选与存储 (Select & Store) ---

    // 1. 生成 Mask
    const vbool16_t v_mask = __riscv_vmflt_vf_f32m2_b16(v_dist, th, vl);

    // 2. 统计有效数量
    long active_count = __riscv_vcpop_m_b16(v_mask, vl);

    if (active_count > 0) {
      // 1. 存储 Inliers (uint32_t)
      // m2 类型的索引压缩后依然是 m2
      vuint32m2_t v_idx_compressed = __riscv_vcompress_vm_u32m2(v_idx, v_mask, vl);
      __riscv_vse32_v_u32m2((uint32_t*)(inliers_out_ptr + nr_p), v_idx_compressed, active_count);

      // 2. 存储 Distances (float -> double)
      // a. 在 m2 级别进行压缩
      vfloat32m2_t v_dist_compressed = __riscv_vcompress_vm_f32m2(v_dist, v_mask, vl);

      // b. 由 m2 宽化为 m4。
      // 这是 RVV 硬件定义的 SEW*2 必须对应 LMUL*2 的要求
      vfloat64m4_t v_dist_double = __riscv_vfwcvt_f_f_v_f64m4(v_dist_compressed, active_count);

      // c. 使用 vse64 直接写回 double 数组
      __riscv_vse64_v_f64m4(dists_out_ptr + nr_p, v_dist_double, active_count);

      nr_p += active_count;
  }

    i += vl;
  }

  // 返回内点总数
  return nr_p;
}
#endif

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

    const vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, y)), v_off_pt, vl);
    const vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, z)), v_off_pt, vl);

    // Byte offsets for PointNT
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    const vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    const vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);

    const vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, curvature)), v_off_norm, vl);

    // Calculate Euclidean distance using the math kernel helper.
    // Data is already in registers, avoiding re-fetching.
    const vfloat32m2_t v_d_euc = pcl::SampleConsensusModelPlane<PointT>::distRVV_f32m2(v_px, v_py, v_pz, v_a, v_b, v_c, v_d, vl);

    // Calculate the acute angle between point normal and plane normal.
    // (Assumes getAcuteAngle3DRVV is implemented similarly to the SSE/AVX versions)
    const vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV_f32m2(v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    // Calculate weight: weight = w_scalar * (1.0 - curvature)
    const vfloat32m2_t v_w = __riscv_vfmul_vf_f32m2(
                          __riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl), w_scalar, vl);

    // Calculate final distance: dist = w * d_norm + (1 - w) * d_euc
    vfloat32m2_t v_dist = __riscv_vfmacc_vv_f32m2(
                            __riscv_vfmul_vv_f32m2(v_w, v_d_norm, vl),
                            __riscv_vfrsub_vf_f32m2(v_w, 1.0f, vl), v_d_euc, vl);
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

  if (!isModelValid (model_coefficients))
  {
    distances.clear ();
    return;
  }

  distances.resize (indices_->size ());

#if defined (__RVV10__)
  // RVV 优化版本
  getDistancesToModelRVV(model_coefficients, distances);
#else
  // 标准版本
  getDistancesToModelStandard(model_coefficients, distances, 0);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::getDistancesToModelStandard (
      const Eigen::VectorXf &model_coefficients, std::vector<double> &distances, std::size_t i) const
{
  // Obtain the plane normal
  Eigen::Vector4f coeff = model_coefficients;
  coeff[3] = 0.0f;

  distances.resize (indices_->size ());

  // Iterate through the 3d points and calculate the distances from them to the plane
  for (; i < indices_->size (); ++i)
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

//////////////////////////////////////////////////////////////////////////
#if defined (__RVV10__)
/** \brief RVV implementation of \c getDistancesToModel: one distance per index in \c indices_, dense write-back.
 *
 * \note Precision: same pipeline as \ref selectWithinDistanceRVV — all distance arithmetic in \c float
 *       (\c vfloat32m2_t), then \c vfwcvt to \c double for \c std::vector<double>. The scalar path
 *       (\ref getDistancesToModelStandard) uses \c double for dot products and \c getAngle3D. Angular
 *       and plane terms therefore may not match bit-for-bit.
 */
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelNormalPlane<PointT, PointNT>::getDistancesToModelRVV (
      const Eigen::VectorXf &model_coefficients, std::vector<double> &distances) const
{
  // 确保 distances 大小正确
  distances.resize(indices_->size());

  // 1. 准备数据指针
  // 转为 uint8_t* 以便后续方便地进行字节偏移计算 (reinterpret_cast)
  const uint8_t* points_base = reinterpret_cast<const uint8_t*>(input_->points.data());
  const uint8_t* normals_base = reinterpret_cast<const uint8_t*>(normals_->points.data());
  const uint8_t* indices_base = reinterpret_cast<const uint8_t*>(indices_->data());

  // 输出指针 (直接写入 vector<double>)
  double* dists_out_ptr = distances.data();
  size_t n_points = indices_->size();

  // 2. 准备系数（float 形式，后续在 float RVV 中计算，在写回时再转 double）
  const float a = model_coefficients[0];
  const float b = model_coefficients[1];
  const float c = model_coefficients[2];
  const float d = model_coefficients[3];
  const float w_scalar = static_cast<float>(normal_distance_weight_);

  // 3. 循环处理
  for (size_t i = 0; i < n_points; )
  {
    // A. 设置 VL
    size_t vl = __riscv_vsetvl_e32m2(n_points - i);

    // B. 加载索引
    // indices_ 是 int vector，这里作为 u32 加载
    vuint32m2_t v_idx = __riscv_vle32_v_u32m2(reinterpret_cast<const uint32_t*>(indices_base + i * 4), vl);

    // 计算结构体的字节偏移量 (Byte Offsets)
    vuint32m2_t v_off_pt = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointT), vl);
    vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);


    // --- C. 加载数据 ---

    // const vfloat32m2x3_t v_xyz = __riscv_vluxseg3ei32_v_f32m2x3(
    //   reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    // const vfloat32m2_t v_px = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
    // const vfloat32m2_t v_py = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
    // const vfloat32m2_t v_pz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

    // const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);
    // const vfloat32m2x3_t v_nxyz = __riscv_vluxseg3ei32_v_f32m2x3(
    //     reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    // const vfloat32m2_t v_nx = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 0);
    // const vfloat32m2_t v_ny = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 1);
    // const vfloat32m2_t v_nz = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 2);

    // C1. 加载 PointT (x, y, z)
    vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, y)), v_off_pt, vl);
    vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, z)), v_off_pt, vl);

    // C2. 加载 PointNT
    vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);

    // C2.2 加载曲率 (curvature)
    const float* curv_base_ptr = reinterpret_cast<const float*>(normals_base + offsetof(PointNT, curvature));
    vfloat32m2_t v_curv = __riscv_vluxei32_v_f32m2(curv_base_ptr, v_off_norm, vl);

    // --- D. 广播系数 ---
    const vfloat32m2_t v_a = __riscv_vfmv_v_f_f32m2(a, vl);
    const vfloat32m2_t v_b = __riscv_vfmv_v_f_f32m2(b, vl);
    const vfloat32m2_t v_c = __riscv_vfmv_v_f_f32m2(c, vl);
    const vfloat32m2_t v_d = __riscv_vfmv_v_f_f32m2(d, vl);

    // --- E. 核心计算（float 域，写回时转 double） ---

    // 1. 欧氏距离：|ax + by + cz + d|
    vfloat32m2_t v_d_euc = pcl::SampleConsensusModelPlane<PointT>::distRVV_f32m2(
        v_px, v_py, v_pz, v_a, v_b, v_c, v_d, vl);
    v_d_euc = __riscv_vfsgnjx_vv_f32m2(v_d_euc, v_d_euc, vl);

    // 2. 夹角距离
    vfloat32m2_t v_d_norm = pcl::getAcuteAngle3DRVV_f32m2(
        v_nx, v_ny, v_nz, v_a, v_b, v_c, vl);

    // 取锐角限制：min(|angle|, PI - |angle|)
    v_d_norm = __riscv_vfsgnjx_vv_f32m2(v_d_norm, v_d_norm, vl);
    vfloat32m2_t v_pi_minus = __riscv_vfrsub_vf_f32m2(v_d_norm, static_cast<float>(M_PI), vl);
    v_d_norm = __riscv_vfmin_vv_f32m2(v_d_norm, v_pi_minus, vl);

    // 3. 权重：weight = w * (1 - curvature)
    vfloat32m2_t v_weight = __riscv_vfmul_vf_f32m2(__riscv_vfrsub_vf_f32m2(v_curv, 1.0f, vl), w_scalar, vl);

    // 4. 最终距离组合：dist = | w * d_norm + (1 - w) * d_euc |
    vfloat32m2_t v_term1 = __riscv_vfmul_vv_f32m2(v_weight, v_d_norm, vl);
    vfloat32m2_t v_one_minus_weight = __riscv_vfrsub_vf_f32m2(v_weight, 1.0f, vl);
    vfloat32m2_t v_term2 = __riscv_vfmul_vv_f32m2(v_one_minus_weight, v_d_euc, vl);
    vfloat32m2_t v_final_dist_f = __riscv_vfadd_vv_f32m2(v_term1, v_term2, vl);
    v_final_dist_f = __riscv_vfsgnjx_vv_f32m2(v_final_dist_f, v_final_dist_f, vl); // __riscv_vfabs_v_f32m2(v_final_dist_f, vl);

    // 5. 写回：float → double
    // 遵循 SEW*2 对应 LMUL*2 的原则
    vfloat64m4_t v_final_dist_d = __riscv_vfwcvt_f_f_v_f64m4(v_final_dist_f, vl);
    __riscv_vse64_v_f64m4(dists_out_ptr + i, v_final_dist_d, vl);

    i += vl;
  }
}
#endif

#define PCL_INSTANTIATE_SampleConsensusModelNormalPlane(PointT, PointNT) template class PCL_EXPORTS pcl::SampleConsensusModelNormalPlane<PointT, PointNT>;

#endif    // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_NORMAL_PLANE_H_

