/*
 * centroid.hpp 单元测试（聚焦 RVV 优化覆盖函数）
 *
 * 标注约定：
 * - 「上游复制」：自 test/common/test_centroid.cpp 复制的 compute3DCentroid、computeCovarianceMatrix（给定质心）与
 *              computeCovarianceMatrixNormalized、computeDemeanedCovariance（关于原点二阶矩）、computeMeanAndCovariance 等回归片段，逻辑与断言尽量保持一致。
 * - 「test-rvv 自建」：轻量用例或 n>=16 的 compute3DCentroid / computeMeanAndCovarianceMatrix / computeCovarianceMatrix（给定质心或关于原点）补充测试（命中对应 RVV 向量路径）。
 * - bun0.pcd：与上游 test/common/test_centroid.cpp 一致，由 main 中 pcl::io::loadPCDFile 读入
 *   pcl::PCLPointCloud2 cloud_blob；用例中 fromPCLPointCloud2 再转为点云（链接 libpcl_io、libpng、zlib）。
 */

#include <pcl/test/gtest.h>
#include <pcl/pcl_tests.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/cloud_iterator.h>

#include <pcl/common/centroid.h>

#include <Eigen/Core>

#include <cstdlib>
#include <iostream>
#include <limits>

using namespace pcl;
using pcl::test::EXPECT_EQ_VECTORS;

pcl::PCLPointCloud2 cloud_blob;

// ---------- 小立方体工具点云（供多个 TEST 复用；非 gtest 用例） ----------
using CubePointCloud = pcl::PointCloud<pcl::PointXYZ>;

// 八个角点坐标 ∈ {-1,1}^3，is_dense=true。
static CubePointCloud
makeCubeCloud ()
{
  CubePointCloud cloud;
  pcl::PointXYZ point;
  for (point.x = -1.0f; point.x < 2.0f; point.x += 2.0f)
    for (point.y = -1.0f; point.y < 2.0f; point.y += 2.0f)
      for (point.z = -1.0f; point.z < 2.0f; point.z += 2.0f)
        cloud.push_back (point);
  cloud.is_dense = true;
  return cloud;
}

/** 8 个角点各重复一次 → 16 点稠密云；均值与协方差与 \a makeCubeCloud 相同（用于触发 RVV 的 n>=16 门限）。 */
static CubePointCloud
makeCubeCloudDup16 ()
{
  CubePointCloud cloud = makeCubeCloud ();
  const std::size_t n8 = cloud.size ();
  for (std::size_t i = 0; i < n8; ++i)
    cloud.push_back (cloud[i]);
  cloud.is_dense = true;
  return cloud;
}

// [上游复制] compute3DCentroid，质心为 Eigen::Vector4f：空/非稠密云、仅 NaN、稠密 8 点立方体、indices 四点子集、非稠密含 NaN 与 PointIndices 等边界；稠密 n=8 时 RVV 内回退标量。
TEST (PCL, compute3DCentroidFloat)
{
  [[maybe_unused]] pcl::PointIndices pindices;
  Indices indices;
  PointXYZ point;
  PointCloud<PointXYZ> cloud;
  Eigen::Vector4f centroid = Eigen::Vector4f::Random ();
  const Eigen::Vector4f old_centroid = centroid;

  // test empty cloud which is dense
  cloud.is_dense = true;
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test empty cloud non_dense
  cloud.is_dense = false;
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test non-empty cloud non_dense (with only invalid points)
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test non-empty cloud non_dense (with only invalid points)
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  indices.push_back (1);
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  cloud.clear ();
  indices.clear ();
  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  // eight points with (0, 0, 0) as centroid and covarmat (1, 0, 0, 0, 1, 0, 0, 0, 1)
  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;

  EXPECT_EQ (compute3DCentroid (cloud, centroid), 8);
  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 0);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (centroid[3], 1);

  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;
  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 4);

  EXPECT_EQ (centroid[0], 0.0);
  EXPECT_EQ (centroid[1], 1.0);
  EXPECT_EQ (centroid[2], 0.0);
  EXPECT_EQ (centroid[3], 1.0);

  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  cloud.is_dense = false;

  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 8);

  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 0);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (centroid[3], 1);

  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  indices.push_back (8); // add the NaN
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 4);

  EXPECT_EQ (centroid[0], 0.0);
  EXPECT_EQ (centroid[1], 1.0);
  EXPECT_EQ (centroid[2], 0.0);
  EXPECT_EQ (centroid[3], 1.0);

  pindices.indices = indices;
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 4);

  EXPECT_EQ (centroid[0], 0.0);
  EXPECT_EQ (centroid[1], 1.0);
  EXPECT_EQ (centroid[2], 0.0);
  EXPECT_EQ (centroid[3], 1.0);
}

// [上游复制] 与 compute3DCentroidFloat 相同场景，质心为 Eigen::Vector4d（双精度输出）。
TEST (PCL, compute3DCentroidDouble)
{
  [[maybe_unused]] pcl::PointIndices pindices;
  Indices indices;
  PointXYZ point;
  PointCloud<PointXYZ> cloud;
  Eigen::Vector4d centroid = Eigen::Vector4d::Random ();
  const Eigen::Vector4d old_centroid = centroid;

  // test empty cloud which is dense
  cloud.is_dense = true;
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test empty cloud non_dense
  cloud.is_dense = false;
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test non-empty cloud non_dense (with only invalid points)
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test non-empty cloud non_dense (with only invalid points)
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  indices.push_back (1);
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 0);
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  cloud.clear ();
  indices.clear ();
  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  // eight points with (0, 0, 0) as centroid and covarmat (1, 0, 0, 0, 1, 0, 0, 0, 1)
  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;

  EXPECT_EQ (compute3DCentroid (cloud, centroid), 8);
  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 0);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (centroid[3], 1);

  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;
  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 4);

  EXPECT_EQ (centroid[0], 0.0);
  EXPECT_EQ (centroid[1], 1.0);
  EXPECT_EQ (centroid[2], 0.0);

  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  cloud.is_dense = false;

  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;
  EXPECT_EQ (compute3DCentroid (cloud, centroid), 8);

  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 0);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (centroid[3], 1);

  centroid[0] = -100;
  centroid[1] = -200;
  centroid[2] = -300;
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  indices.push_back (8); // add the NaN
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 4);

  EXPECT_EQ (centroid[0], 0.0);
  EXPECT_EQ (centroid[1], 1.0);
  EXPECT_EQ (centroid[2], 0.0);
  EXPECT_EQ (centroid[3], 1.0);

  pindices.indices = indices;
  EXPECT_EQ (compute3DCentroid (cloud, indices, centroid), 4);

  EXPECT_EQ (centroid[0], 0.0);
  EXPECT_EQ (centroid[1], 1.0);
  EXPECT_EQ (centroid[2], 0.0);
  EXPECT_EQ (centroid[3], 1.0);
}

// [上游复制] ConstCloudIterator 版 compute3DCentroid：indices 子集与全云迭代、float/double 质心；非稠密全云迭代跳过 NaN、仅含 NaN 的 indices 返回 0 且不改正质心。
TEST (PCL, compute3DCentroidCloudIterator)
{
  Indices indices;
  PointXYZ point;
  PointCloud<PointXYZ> cloud;
  Eigen::Vector4f centroid_f;

  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;

  // Test finite data
  {
    ConstCloudIterator<PointXYZ> it (cloud, indices);

    EXPECT_EQ (compute3DCentroid (it, centroid_f), 4);

    EXPECT_EQ (centroid_f[0], 0.0f);
    EXPECT_EQ (centroid_f[1], 1.0f);
    EXPECT_EQ (centroid_f[2], 0.0f);
    EXPECT_EQ (centroid_f[3], 1.0f);

    Eigen::Vector4d centroid_d;
    it.reset ();
    EXPECT_EQ (compute3DCentroid (it, centroid_d), 4);

    EXPECT_EQ (centroid_d[0], 0.0);
    EXPECT_EQ (centroid_d[1], 1.0);
    EXPECT_EQ (centroid_d[2], 0.0);
    EXPECT_EQ (centroid_d[3], 1.0);
  }

  // Test for non-finite data
  {
    point.getVector3fMap () << std::numeric_limits<float>::quiet_NaN (),
        std::numeric_limits<float>::quiet_NaN (),
        std::numeric_limits<float>::quiet_NaN ();
    cloud.push_back (point);
    cloud.is_dense = false;
    ConstCloudIterator<PointXYZ> it (cloud);

    EXPECT_EQ (8, compute3DCentroid (it, centroid_f));
    EXPECT_EQ_VECTORS (Eigen::Vector4f (0.f, 0.f, 0.f, 1.f), centroid_f);

    const Eigen::Vector4f old_centroid = centroid_f;
    indices.clear ();
    indices.push_back (cloud.size () - 1);
    ConstCloudIterator<PointXYZ> it2 (cloud, indices);
    // zero valid points and centroid remains unchanged
    EXPECT_EQ (0, compute3DCentroid (it2, centroid_f));
    EXPECT_EQ (old_centroid, centroid_f);
  }
}

// [test-rvv 自建] 稠密小立方体（8 点）整云质心，期望值 (0,0,0,1)；n<16，不跑 RVV 向量循环。
TEST (PCL, compute3DCentroid_dense)
{
  CubePointCloud cloud = makeCubeCloud ();
  Eigen::Vector4f centroid;
  const auto n = pcl::compute3DCentroid (cloud, centroid);
  EXPECT_EQ (n, 8u);
  EXPECT_FLOAT_EQ (centroid[0], 0.0f);
  EXPECT_FLOAT_EQ (centroid[1], 0.0f);
  EXPECT_FLOAT_EQ (centroid[2], 0.0f);
  EXPECT_FLOAT_EQ (centroid[3], 1.0f);
}

// [test-rvv 自建] 同上立方体，仅用 4 个索引（y=1 平面四角），期望质心 y=1；n=4，RVV 内回退标量。
TEST (PCL, compute3DCentroid_indices)
{
  CubePointCloud cloud = makeCubeCloud ();
  pcl::Indices indices = {2, 3, 6, 7}; // y=1 平面
  Eigen::Vector4f centroid;
  const auto n = pcl::compute3DCentroid (cloud, indices, centroid);
  EXPECT_EQ (n, 4u);
  EXPECT_FLOAT_EQ (centroid[0], 0.0f);
  EXPECT_FLOAT_EQ (centroid[1], 1.0f);
  EXPECT_FLOAT_EQ (centroid[2], 0.0f);
  EXPECT_FLOAT_EQ (centroid[3], 1.0f);
}

// [test-rvv 自建] makeCubeCloudDup16（16 点稠密）整云质心；n>=16 且 PointXYZ 时走 compute3DCentroidRVV 向量路径。
TEST (PCL, compute3DCentroid_dense_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  ASSERT_EQ (cloud.size (), 16u);
  Eigen::Vector4f centroid;
  const auto n = pcl::compute3DCentroid (cloud, centroid);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (centroid[0], 0.0f);
  EXPECT_FLOAT_EQ (centroid[1], 0.0f);
  EXPECT_FLOAT_EQ (centroid[2], 0.0f);
  EXPECT_FLOAT_EQ (centroid[3], 1.0f);
}

// [test-rvv 自建] 16 点云 + indices 0..15；与整云等价，覆盖 indices 版 compute3DCentroidRVV（n>=16）。
TEST (PCL, compute3DCentroid_indices_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  pcl::Indices indices (16);
  for (int i = 0; i < 16; ++i)
    indices[static_cast<std::size_t> (i)] = i;
  Eigen::Vector4f centroid;
  const auto n = pcl::compute3DCentroid (cloud, indices, centroid);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (centroid[0], 0.0f);
  EXPECT_FLOAT_EQ (centroid[1], 0.0f);
  EXPECT_FLOAT_EQ (centroid[2], 0.0f);
  EXPECT_FLOAT_EQ (centroid[3], 1.0f);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// [上游复制] test/common/test_centroid.cpp — computeCovarianceMatrix（给定质心），含空/NaN/indices 子集与非稠密分支。
TEST (PCL, computeCovarianceMatrix)
{
  PointCloud<PointXYZ> cloud;
  PointXYZ point;
  Indices indices;
  Eigen::Vector4f centroid;
  Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Random ();
  const Eigen::Matrix3f old_covariance_matrix = covariance_matrix;

  centroid[0] = 0;
  centroid[1] = 0;
  centroid[2] = 0;

  // test empty cloud which is dense
  cloud.is_dense = true;
  EXPECT_EQ (computeCovarianceMatrix (cloud, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test empty cloud non_dense
  cloud.is_dense = false;
  EXPECT_EQ (computeCovarianceMatrix (cloud, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test non-empty cloud non_dense (with only invalid points)
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  EXPECT_EQ (computeCovarianceMatrix (cloud, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test non-empty cloud non_dense (with only invalid points)
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  indices.push_back (1);
  EXPECT_EQ (computeCovarianceMatrix (cloud, indices, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  cloud.clear ();
  indices.clear ();
  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  // eight points with (0, 0, 0) as centroid and covarmat (1, 0, 0, 0, 1, 0, 0, 0, 1)

  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[0] = 0;
  centroid[1] = 0;
  centroid[2] = 0;

  EXPECT_EQ (computeCovarianceMatrix (cloud, centroid, covariance_matrix), 8);
  EXPECT_EQ (covariance_matrix (0, 0), 8);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 8);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 8);

  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[1] = 1;

  EXPECT_EQ (computeCovarianceMatrix (cloud, indices, centroid, covariance_matrix), 4);
  EXPECT_EQ (covariance_matrix (0, 0), 4);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 0);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 4);

  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  cloud.is_dense = false;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[1] = 0;

  EXPECT_EQ (computeCovarianceMatrix (cloud, centroid, covariance_matrix), 8);
  EXPECT_EQ (covariance_matrix (0, 0), 8);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 8);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 8);

  indices.push_back (8); // add the NaN
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[1] = 1;

  EXPECT_EQ (computeCovarianceMatrix (cloud, indices, centroid, covariance_matrix), 4);
  EXPECT_EQ (covariance_matrix (0, 0), 4);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 0);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 4);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// [上游复制] test/common/test_centroid.cpp — computeCovarianceMatrixNormalized（给定质心）。
TEST (PCL, computeCovarianceMatrixNormalized)
{
  PointCloud<PointXYZ> cloud;
  PointXYZ point;
  Indices indices;
  Eigen::Vector4f centroid;
  Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Random ();
  const Eigen::Matrix3f old_covariance_matrix = covariance_matrix;

  centroid[0] = 0;
  centroid[1] = 0;
  centroid[2] = 0;

  // test empty cloud which is dense
  cloud.is_dense = true;
  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test empty cloud non_dense
  cloud.is_dense = false;
  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test non-empty cloud non_dense
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test non-empty cloud non_dense
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  indices.push_back (1);
  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, indices, centroid, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  cloud.clear ();
  indices.clear ();
  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  // eight points with (0, 0, 0) as centroid and covarmat (1, 0, 0, 0, 1, 0, 0, 0, 1)

  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[0] = 0;
  centroid[1] = 0;
  centroid[2] = 0;

  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, centroid, covariance_matrix), 8);

  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[1] = 1;

  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, indices, centroid, covariance_matrix), 4);

  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 0);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  cloud.is_dense = false;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[1] = 0;

  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, centroid, covariance_matrix), 8);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  indices.push_back (8); // add the NaN
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[1] = 1;

  EXPECT_EQ (computeCovarianceMatrixNormalized (cloud, indices, centroid, covariance_matrix), 4);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 0);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// [上游复制] test/common/test_centroid.cpp — computeCovarianceMatrix（关于原点二阶矩 / n），套件名同上游 computeDemeanedCovariance。
TEST (PCL, computeDemeanedCovariance)
{
  PointCloud<PointXYZ> cloud;
  PointXYZ point;
  Indices indices;
  Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Random ();
  const Eigen::Matrix3f old_covariance_matrix = covariance_matrix;

  // test empty cloud which is dense
  cloud.is_dense = true;
  EXPECT_EQ (computeCovarianceMatrix (cloud, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test empty cloud non_dense
  cloud.is_dense = false;
  EXPECT_EQ (computeCovarianceMatrix (cloud, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test non-empty cloud non_dense
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  EXPECT_EQ (computeCovarianceMatrix (cloud, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  // test non-empty cloud non_dense
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  indices.push_back (1);
  EXPECT_EQ (computeCovarianceMatrix (cloud, indices, covariance_matrix), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged

  cloud.clear ();
  indices.clear ();

  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  // eight points with (0, 0, 0) as centroid and covarmat (1, 0, 0, 0, 1, 0, 0, 0, 1)

  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;

  EXPECT_EQ (computeCovarianceMatrix (cloud, covariance_matrix), 8);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;

  EXPECT_EQ (computeCovarianceMatrix (cloud, indices, covariance_matrix), 4);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  cloud.is_dense = false;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;

  EXPECT_EQ (computeCovarianceMatrix (cloud, covariance_matrix), 8);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  indices.push_back (8); // add the NaN
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;

  EXPECT_EQ (computeCovarianceMatrix (cloud, indices, covariance_matrix), 4);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);
}

// [test-rvv 自建] 16 点稠密：给定质心为原点时 computeCovarianceMatrix；n>=16 且 PointXYZ 时走 computeCovarianceMatrixCentroidRVV。
TEST (PCL, computeCovarianceMatrix_given_centroid_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  const auto n = pcl::computeCovarianceMatrix (cloud, centroid, cov);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (cov (0, 0), 16.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 16.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 16.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

// [test-rvv 自建] indices 全量 16 点：与上一致，命中给定质心 + indices 的 RVV 路径。
TEST (PCL, computeCovarianceMatrix_given_centroid_indices_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Indices indices;
  for (std::size_t i = 0; i < cloud.size (); ++i)
    indices.push_back (static_cast<int> (i));
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  const auto n = pcl::computeCovarianceMatrix (cloud, indices, centroid, cov);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (cov (0, 0), 16.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 16.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 16.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

// [test-rvv 自建] computeCovarianceMatrixNormalized：给定质心路径上 raw 协方差和再 /n；Dup16 且质心原点 → 对角为 1。
TEST (PCL, computeCovarianceMatrixNormalized_given_centroid_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  const auto n = pcl::computeCovarianceMatrixNormalized (cloud, centroid, cov);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (cov (0, 0), 1.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 1.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 1.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

// [test-rvv 自建] 关于原点二阶矩 / n；Dup16 稠密整云，命中 computeCovarianceMatrixOriginRVV。
TEST (PCL, computeCovarianceMatrix_aboutOrigin_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  const auto n = pcl::computeCovarianceMatrix (cloud, cov);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (cov (0, 0), 1.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 1.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 1.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

// [test-rvv 自建] indices 全量 16 点，同上，命中 indices 版 about-origin RVV。
TEST (PCL, computeCovarianceMatrix_aboutOrigin_indices_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Indices indices;
  for (std::size_t i = 0; i < cloud.size (); ++i)
    indices.push_back (static_cast<int> (i));
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  const auto n = pcl::computeCovarianceMatrix (cloud, indices, cov);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (cov (0, 0), 1.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 1.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 1.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// [上游复制] computeMeanAndCovarianceMatrix：空/非稠密/NaN、稠密 8 点协方差与 indices 子集、非稠密跳过 NaN（见 test/common/test_centroid.cpp）；稠密 8 点段不调用 RVV（n<16）。
TEST (PCL, computeMeanAndCovariance)
{
  PointCloud<PointXYZ> cloud;
  PointXYZ point;
  Indices indices;
  Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Random ();
  Eigen::Vector4f centroid = Eigen::Vector4f::Random ();
  const Eigen::Matrix3f old_covariance_matrix = covariance_matrix;
  const Eigen::Vector4f old_centroid = centroid;

  // test empty cloud which is dense
  cloud.is_dense = true;
  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, covariance_matrix, centroid), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test empty cloud non_dense
  cloud.is_dense = false;
  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, covariance_matrix, centroid), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test non-empty cloud non_dense
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, covariance_matrix, centroid), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  // test non-empty cloud non_dense
  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  indices.push_back (1);
  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix, centroid), 0);
  EXPECT_EQ (old_covariance_matrix, covariance_matrix); // cov. matrix remains unchanged
  EXPECT_EQ (old_centroid, centroid); // centroid remains unchanged

  cloud.clear ();
  indices.clear ();

  for (point.x = -1; point.x < 2; point.x += 2)
  {
    for (point.y = -1; point.y < 2; point.y += 2)
    {
      for (point.z = -1; point.z < 2; point.z += 2)
      {
        cloud.push_back (point);
      }
    }
  }
  cloud.is_dense = true;

  // eight points with (0, 0, 0) as centroid and covarmat (1, 0, 0, 0, 1, 0, 0, 0, 1)

  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[0] = -100;
  centroid[1] = -101;
  centroid[2] = -102;
  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, covariance_matrix, centroid), 8);

  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 0);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  indices.resize (4); // only positive y values
  indices[0] = 2;
  indices[1] = 3;
  indices[2] = 6;
  indices[3] = 7;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[0] = -100;
  centroid[1] = -101;
  centroid[2] = -102;

  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix, centroid), 4);
  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 1);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 0);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN ();
  cloud.push_back (point);
  cloud.is_dense = false;
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[0] = -100;
  centroid[1] = -101;
  centroid[2] = -102;

  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, covariance_matrix, centroid), 8);
  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 0);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 1);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);

  indices.push_back (8); // add the NaN
  covariance_matrix << -100, -101, -102, -110, -111, -112, -120, -121, -122;
  centroid[0] = -100;
  centroid[1] = -101;
  centroid[2] = -102;

  EXPECT_EQ (computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix, centroid), 4);
  EXPECT_EQ (centroid[0], 0);
  EXPECT_EQ (centroid[1], 1);
  EXPECT_EQ (centroid[2], 0);
  EXPECT_EQ (covariance_matrix (0, 0), 1);
  EXPECT_EQ (covariance_matrix (0, 1), 0);
  EXPECT_EQ (covariance_matrix (0, 2), 0);
  EXPECT_EQ (covariance_matrix (1, 0), 0);
  EXPECT_EQ (covariance_matrix (1, 1), 0);
  EXPECT_EQ (covariance_matrix (1, 2), 0);
  EXPECT_EQ (covariance_matrix (2, 0), 0);
  EXPECT_EQ (covariance_matrix (2, 1), 0);
  EXPECT_EQ (covariance_matrix (2, 2), 1);
}

// [test-rvv 自建] 16 点稠密（makeCubeCloudDup16）整云均值与协方差；满足 n>=16 时走 computeMeanAndCovarianceMatrixRVV。
TEST (PCL, computeMeanAndCovarianceMatrix_dense_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  ASSERT_EQ (cloud.size (), 16u);
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  Eigen::Vector4f centroid = Eigen::Vector4f::Zero ();
  const auto n = pcl::computeMeanAndCovarianceMatrix (cloud, cov, centroid);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (centroid[0], 0.0f);
  EXPECT_FLOAT_EQ (centroid[1], 0.0f);
  EXPECT_FLOAT_EQ (centroid[2], 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 0), 1.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 1.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 1.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

// [test-rvv 自建] 16 点云 + indices 0..15，与整云等价；覆盖 indices 版 computeMeanAndCovarianceMatrixRVV。
TEST (PCL, computeMeanAndCovarianceMatrix_indices_rvvMinPoints)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  pcl::Indices indices (16);
  for (int i = 0; i < 16; ++i)
    indices[static_cast<std::size_t> (i)] = i;
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero ();
  Eigen::Vector4f centroid = Eigen::Vector4f::Zero ();
  const auto n = pcl::computeMeanAndCovarianceMatrix (cloud, indices, cov, centroid);
  EXPECT_EQ (n, 16u);
  EXPECT_FLOAT_EQ (centroid[0], 0.0f);
  EXPECT_FLOAT_EQ (centroid[1], 0.0f);
  EXPECT_FLOAT_EQ (centroid[2], 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 0), 1.0f);
  EXPECT_FLOAT_EQ (cov (1, 1), 1.0f);
  EXPECT_FLOAT_EQ (cov (2, 2), 1.0f);
  EXPECT_FLOAT_EQ (cov (0, 1), 0.0f);
  EXPECT_FLOAT_EQ (cov (0, 2), 0.0f);
  EXPECT_FLOAT_EQ (cov (1, 2), 0.0f);
}

// [test-rvv 自建] demeanPointCloud：质心已为原点时输出应与输入首点一致（最小回归）。
TEST (PCL, demeanPointCloud_dense)
{
  CubePointCloud cloud = makeCubeCloud ();
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  CubePointCloud cloud_out;
  pcl::demeanPointCloud (cloud, centroid, cloud_out);
  ASSERT_EQ (cloud_out.size (), cloud.size ());
  EXPECT_FLOAT_EQ (cloud_out[0].x, cloud[0].x);
  EXPECT_FLOAT_EQ (cloud_out[0].y, cloud[0].y);
  EXPECT_FLOAT_EQ (cloud_out[0].z, cloud[0].z);
}

// [test-rvv 自建] 16 点稠密云 + 零质心；n>=16 且 PointXYZ 时 demeanPointCloud RVV。
TEST (PCL, demeanPointCloud_dup16_dense)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  CubePointCloud cloud_out;
  pcl::demeanPointCloud (cloud, centroid, cloud_out);
  ASSERT_EQ (cloud_out.size (), cloud.size ());
  EXPECT_FLOAT_EQ (cloud_out[0].x, cloud[0].x);
  EXPECT_FLOAT_EQ (cloud_out[15].z, cloud[15].z);
}

// [test-rvv 自建] indices 全量 0..15；与整云 demean 等价，覆盖 demeanPointCloud RVV（indices 重载）。
TEST (PCL, demeanPointCloud_indices_dup16_dense)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  pcl::Indices indices (16);
  for (int i = 0; i < 16; ++i)
    indices[static_cast<std::size_t> (i)] = i;
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  CubePointCloud cloud_out;
  pcl::demeanPointCloud (cloud, indices, centroid, cloud_out);
  ASSERT_EQ (cloud_out.size (), 16u);
  EXPECT_FLOAT_EQ (cloud_out[0].x, cloud[0].x);
  EXPECT_FLOAT_EQ (cloud_out[15].z, cloud[15].z);
}

// [test-rvv 自建] Eigen 4×N float 输出；n>=16 可走 demeanPointCloudEigen RVV。
TEST (PCL, demeanPointCloud_eigen4x_dyn_dup16_dense)
{
  CubePointCloud cloud = makeCubeCloudDup16 ();
  Eigen::Vector4f centroid (0.0f, 0.0f, 0.0f, 1.0f);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_out;
  pcl::demeanPointCloud (cloud, centroid, mat_out);
  ASSERT_EQ (mat_out.rows (), 4);
  ASSERT_EQ (mat_out.cols (), 16);
  EXPECT_FLOAT_EQ (mat_out (0, 0), cloud[0].x);
  EXPECT_FLOAT_EQ (mat_out (2, 15), cloud[15].z);
}

// [test-rvv 自建] main 已加载 bun0.pcd → cloud_blob；大点云稠密 compute3DCentroid，与上游期望质心近似值对比（n>>16，可走 RVV）。
TEST (PCL, compute3DCentroid_bun0_loadPCDFile)
{
  PointCloud<PointXYZ> cloud;
  fromPCLPointCloud2 (cloud_blob, cloud);
  ASSERT_GT (cloud.size (), 0u);
  Eigen::Vector4f centroid;
  const auto n = compute3DCentroid (cloud, centroid);
  EXPECT_EQ (n, cloud.size ());
  EXPECT_NEAR (centroid[0], -0.0290809, 1e-4);
  EXPECT_NEAR (centroid[1], 0.102653, 1e-4);
  EXPECT_NEAR (centroid[2], 0.027302, 1e-4);
  EXPECT_NEAR (centroid[3], 1.0, 1e-4);
}

// 从 argv[1] 读取 bun0.pcd 到全局 cloud_blob，再交给各用例；与上游 centroid 测试数据源约定一致。
int
main (int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "No test file given. Please pass `bun0.pcd` path (same as upstream test/common/test_centroid.cpp)."
              << std::endl;
    return -1;
  }
  if (io::loadPCDFile (argv[1], cloud_blob) < 0)
  {
    std::cerr << "Failed to read test file: " << argv[1] << std::endl;
    return -1;
  }

  testing::InitGoogleTest (&argc, argv);
  return RUN_ALL_TESTS ();
}
