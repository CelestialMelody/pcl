/*
 * 仅链接 gtest，不链 libpcl_common。卷积实现见同目录 gaussian_convolve_float_local.hpp
 *（与 common/src/gaussian.cpp 内联卷积需人工同步）。
 * 须 -D__RVV10__ 才编译 RVV 路径。
 */
#if !defined(__RVV10__)
#error "标量/RVV 对拍需 -D__RVV10__（与 RVV 工具链）"
#endif

#include "gaussian_convolve_float_local.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
namespace conv = gaussian_convolve_local;

void
fillPattern (std::vector<float> &buf, std::size_t w, std::size_t h)
{
  buf.resize (w * h);
  for (std::size_t j = 0; j < h; ++j)
    for (std::size_t i = 0; i < w; ++i)
      buf[i + j * w] = 0.01f * static_cast<float> ((i * 17u + j * 23u + (i * j) % 7) % 251);
}

void
expectBufNear (const std::vector<float> &a, const std::vector<float> &b, float tol)
{
  ASSERT_EQ (a.size (), b.size ());
  for (std::size_t k = 0; k < a.size (); ++k)
    EXPECT_NEAR (a[k], b[k], tol) << " idx=" << k;
}

/* 一维高斯、奇数长、和为 1；与库内 GaussianKernel 不必一致，对拍只要求 std==RVV。 */
void
makeGauss1D (float sigma, std::size_t max_taps, std::vector<float> &out)
{
  const std::size_t n = (max_taps % 2u == 1u) ? max_taps : (max_taps + 1u);
  const int hw = static_cast<int> (n / 2);
  const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
  out.resize (n);
  float s = 0.f;
  for (int i = 0; i < static_cast<int> (n); ++i)
  {
    const int t = i - hw;
    out[static_cast<std::size_t> (i)] = std::exp (-static_cast<float> (t * t) * inv2s2);
    s += out[static_cast<std::size_t> (i)];
  }
  for (float &v : out)
    v /= s;
}

} // namespace

TEST (PCL, gaussian_float_convolveRows_std_vs_rvv)
{
  std::vector<float> k;
  makeGauss1D (2.0f, 11u, k);

  const std::size_t w = 64, h = 33;
  std::vector<float> in, o_std, o_rvv;
  fillPattern (in, w, h);
  o_std.resize (w * h, 0.f);
  o_rvv.resize (w * h, 0.f);

  conv::convolveRowsStandard (in.data (), o_std.data (), w, h, k.data (), k.size ());
  conv::convolveRowsRVV (in.data (), o_rvv.data (), w, h, k.data (), k.size ());

  expectBufNear (o_std, o_rvv, 1e-4f);
}

TEST (PCL, gaussian_float_convolveCols_std_vs_rvv)
{
  std::vector<float> k;
  makeGauss1D (2.5f, 9u, k);

  const std::size_t w = 37, h = 58;
  std::vector<float> in, o_std, o_rvv;
  fillPattern (in, w, h);
  o_std.resize (w * h, 0.f);
  o_rvv.resize (w * h, 0.f);

  conv::convolveColsStandard (in.data (), o_std.data (), w, h, k.data (), k.size ());
  conv::convolveColsRVV (in.data (), o_rvv.data (), w, h, k.data (), k.size ());

  expectBufNear (o_std, o_rvv, 1e-4f);
}

TEST (PCL, gaussian_float_separable_rows_then_cols_std_vs_rvv)
{
  std::vector<float> k;
  makeGauss1D (1.5f, 7u, k);

  const std::size_t w = 48, h = 40;
  std::vector<float> in, r_std, r_rvv, o_std, o_rvv;
  fillPattern (in, w, h);
  r_std.resize (w * h, 0.f);
  r_rvv.resize (w * h, 0.f);
  o_std.resize (w * h, 0.f);
  o_rvv.resize (w * h, 0.f);

  conv::convolveRowsStandard (in.data (), r_std.data (), w, h, k.data (), k.size ());
  conv::convolveRowsRVV (in.data (), r_rvv.data (), w, h, k.data (), k.size ());
  expectBufNear (r_std, r_rvv, 1e-4f);

  conv::convolveColsStandard (r_std.data (), o_std.data (), w, h, k.data (), k.size ());
  conv::convolveColsRVV (r_rvv.data (), o_rvv.data (), w, h, k.data (), k.size ());

  expectBufNear (o_std, o_rvv, 1e-4f);
}

int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);
  return RUN_ALL_TESTS ();
}
