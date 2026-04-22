/*
 * test-rvv：针对 common/src/gaussian.cpp 中实现的 API 做回归与补充。
 *
 * 覆盖范围（均解析到 GaussianKernel 在 gaussian.cpp 内的非模板定义）：
 *   - compute(sigma, kernel)
 *   - compute(sigma, kernel, derivative)
 *   - convolveRows / convolveCols（PointCloud<float> 专版，与 impl/gaussian.hpp 模板无关）
 *   - input == output 时的别名路径（整云拷贝后再卷积）
 */

#include <pcl/test/gtest.h>
#include <pcl/common/gaussian.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>

#include <cmath>

// ---------- [上游] test/common/test_gaussian.cpp：compute 两版本 golden ----------
TEST (PCL, GaussianKernel_compute_golden)
{
  Eigen::VectorXf kernel (31);
  kernel << 0.000888059f, 0.00158611f, 0.00272177f, 0.00448744f, 0.00710844f, 0.0108188f, 0.0158201f,
      0.0222264f, 0.0300025f, 0.0389112f, 0.0484864f, 0.0580487f, 0.0667719f, 0.0737944f, 0.0783576f,
      0.0799405f, 0.0783576f, 0.0737944f, 0.0667719f, 0.0580487f, 0.0484864f, 0.0389112f, 0.0300025f,
      0.0222264f, 0.0158201f, 0.0108188f, 0.00710844f, 0.00448744f, 0.00272177f, 0.00158611f,
      0.000888059f;

  Eigen::VectorXf derivative (35);
  derivative << 0.000168673f, 0.000307151f, 0.000535285f, 0.000892304f, 0.00142183f, 0.00216388f,
      0.00314209f, 0.00434741f, 0.00572143f, 0.00714516f, 0.00843934f, 0.00938163f, 0.00974186f,
      0.0093305f, 0.00804947f, 0.0059307f, 0.00314871f, 0.0f, -0.00314871f, -0.0059307f, -0.00804947f,
      -0.0093305f, -0.00974186f, -0.00938163f, -0.00843934f, -0.00714516f, -0.00572143f, -0.00434741f,
      -0.00314209f, -0.00216388f, -0.00142183f, -0.000892304f, -0.000535285f, -0.000307151f,
      -0.000168673f;
  pcl::GaussianKernel gk;
  Eigen::VectorXf computed_kernel, computed_derivative;

  gk.compute (5, computed_kernel);
  EXPECT_EQ (kernel.size (), computed_kernel.size ());
  for (Eigen::Index i = 0; i < kernel.size (); ++i)
    EXPECT_NEAR (kernel[i], computed_kernel[i], 1e-4f);

  gk.compute (5, computed_kernel, computed_derivative);
  EXPECT_EQ (kernel.size (), computed_kernel.size ());
  for (Eigen::Index i = 0; i < kernel.size (); ++i)
    EXPECT_NEAR (kernel[i], computed_kernel[i], 1e-4f);
  EXPECT_EQ (derivative.size (), computed_derivative.size ());
  for (Eigen::Index i = 0; i < derivative.size (); ++i)
    EXPECT_NEAR (derivative[i], computed_derivative[i], 1e-4f);
}

// ---------- gaussian.cpp：float 行卷积，常数图 + 归一化核 → 内区 ≈ 1 ----------
TEST (PCL, GaussianKernel_convolveRows_float_constant)
{
  pcl::GaussianKernel gk;
  Eigen::VectorXf kernel;
  gk.compute (3.0f, kernel);

  const int w = 64, h = 32;
  pcl::PointCloud<float> in (w, h), out;
  for (std::size_t i = 0; i < in.size (); ++i)
    in[i] = 1.0f;

  gk.convolveRows (in, kernel, out);

  const int r = static_cast<int> (kernel.size () / 2);
  for (int j = 0; j < h; ++j)
    for (int i = r; i < w - r; ++i)
      EXPECT_NEAR (out (i, j), 1.0f, 1e-4f) << " i=" << i << " j=" << j;
}

// ---------- gaussian.cpp：float 列卷积 ----------
TEST (PCL, GaussianKernel_convolveCols_float_constant)
{
  pcl::GaussianKernel gk;
  Eigen::VectorXf kernel;
  gk.compute (3.0f, kernel);

  const int w = 48, h = 72;
  pcl::PointCloud<float> in (w, h), out;
  for (std::size_t i = 0; i < in.size (); ++i)
    in[i] = 1.0f;

  gk.convolveCols (in, kernel, out);

  const int r = static_cast<int> (kernel.size () / 2);
  for (int j = r; j < h - r; ++j)
    for (int i = 0; i < w; ++i)
      EXPECT_NEAR (out (i, j), 1.0f, 1e-4f) << " i=" << i << " j=" << j;
}

static float
refRowConv (const pcl::PointCloud<float> &in, const Eigen::VectorXf &kernel, int i, int j)
{
  const int r = static_cast<int> (kernel.size () / 2);
  float s = 0.f;
  for (int k = 0; k < kernel.size (); ++k)
  {
    const int l = i - r + k;
    s += in (l, j) * kernel[k];
  }
  return s;
}

TEST (PCL, GaussianKernel_convolveRows_float_reference)
{
  pcl::GaussianKernel gk;
  Eigen::VectorXf kernel;
  gk.compute (2.5f, kernel);

  const int w = 128, h = 16;
  pcl::PointCloud<float> in (w, h), out;
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
      in (i, j) = 0.01f * static_cast<float> ((i + j * 13) % 97);

  gk.convolveRows (in, kernel, out);

  const int r = static_cast<int> (kernel.size () / 2);
  for (int j = 0; j < h; ++j)
    for (int i = r; i < w - r; i += 7)
    {
      const float ref = refRowConv (in, kernel, i, j);
      EXPECT_NEAR (out (i, j), ref, 1e-4f) << " i=" << i << " j=" << j;
    }
}

// ---------- gaussian.cpp：&input == &output 别名分支（先拷贝再卷积） ----------
TEST (PCL, GaussianKernel_convolveRows_inplace_alias)
{
  pcl::GaussianKernel gk;
  Eigen::VectorXf kernel;
  gk.compute (2.0f, kernel);

  const int w = 32, h = 24;
  pcl::PointCloud<float> io (w, h);
  for (int j = 0; j < h; ++j)
    for (int i = 0; i < w; ++i)
      io (i, j) = static_cast<float> (i + j);

  pcl::PointCloud<float> ref (w, h);
  gk.convolveRows (io, kernel, ref);

  gk.convolveRows (io, kernel, io);

  for (std::size_t k = 0; k < io.size (); ++k)
    EXPECT_NEAR (io[k], ref[k], 1e-4f) << " k=" << k;
}

int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);
  return RUN_ALL_TESTS ();
}
