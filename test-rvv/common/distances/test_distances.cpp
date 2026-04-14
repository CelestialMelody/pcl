/*
 * distances.h 单元测试（覆盖常用 API 的基本正确性）
 */

#include <pcl/test/gtest.h>

#include <pcl/common/distances.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

TEST(PCL, squaredEuclideanDistance_PointXYZ)
{
  PointT a, b;
  a.x = 1.0f; a.y = 2.0f; a.z = 3.0f;
  b.x = 4.0f; b.y = 6.0f; b.z = 3.0f;
  const float d2 = pcl::squaredEuclideanDistance(a, b);
  EXPECT_FLOAT_EQ(d2, 25.0f);
}

TEST(PCL, euclideanDistance_PointXYZ)
{
  PointT a, b;
  a.x = 0.0f; a.y = 0.0f; a.z = 0.0f;
  b.x = 0.0f; b.y = 3.0f; b.z = 4.0f;
  const float d = pcl::euclideanDistance(a, b);
  EXPECT_FLOAT_EQ(d, 5.0f);
}

TEST(PCL, sqrPointToLineDistance_basic)
{
  const Eigen::Vector4f pt(1.0f, 0.0f, 0.0f, 0.0f);
  const Eigen::Vector4f line_pt(0.0f, 0.0f, 0.0f, 0.0f);
  const Eigen::Vector4f line_dir(1.0f, 1.0f, 0.0f, 0.0f);
  const double d = std::sqrt(pcl::sqrPointToLineDistance(pt, line_pt, line_dir));
  EXPECT_NEAR(d, std::sqrt(2.0) / 2.0, 1e-4);
}

TEST(PCL, getMaxSegment_cloud)
{
  PointCloud cloud;
  cloud.resize(4);
  cloud.is_dense = true;

  cloud[0].x = 0.0f; cloud[0].y = 0.0f; cloud[0].z = 0.0f;
  cloud[1].x = 3.0f; cloud[1].y = 0.0f; cloud[1].z = 0.0f;
  cloud[2].x = 0.0f; cloud[2].y = 4.0f; cloud[2].z = 0.0f;
  cloud[3].x = 1.0f; cloud[3].y = 1.0f; cloud[3].z = 0.0f;

  PointT pmin, pmax;
  const double len = pcl::getMaxSegment(cloud, pmin, pmax);
  EXPECT_NEAR(len, 5.0, 1e-6);

  const auto is_endpoint = [&](const PointT& p) {
    return (p.x == 3.0f && p.y == 0.0f && p.z == 0.0f) ||
           (p.x == 0.0f && p.y == 4.0f && p.z == 0.0f);
  };

  // 端点顺序不保证，仅检查返回点属于最远点对之一
  EXPECT_TRUE(is_endpoint(pmin));
  EXPECT_TRUE(is_endpoint(pmax));
}

TEST(PCL, getMaxSegment_indices)
{
  PointCloud cloud;
  cloud.resize(5);
  cloud.is_dense = true;

  cloud[0].x = 0.0f; cloud[0].y = 0.0f; cloud[0].z = 0.0f;
  cloud[1].x = 3.0f; cloud[1].y = 0.0f; cloud[1].z = 0.0f;
  cloud[2].x = 0.0f; cloud[2].y = 4.0f; cloud[2].z = 0.0f;
  cloud[3].x = 10.0f; cloud[3].y = 0.0f; cloud[3].z = 0.0f;
  cloud[4].x = 0.0f; cloud[4].y = 10.0f; cloud[4].z = 0.0f;

  // 子集：仅包含前三个点，则最远距离仍应为 5
  pcl::Indices indices;
  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(2);

  PointT pmin, pmax;
  const double len = pcl::getMaxSegment(cloud, indices, pmin, pmax);
  EXPECT_NEAR(len, 5.0, 1e-6);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

