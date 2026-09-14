/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2026-, Open Perception Inc.
 *
 *  All rights reserved
 */

#include <pcl/features/fpfh.h>
#include <pcl/test/gtest.h>
#include <pcl/point_types.h>

#include <array>
#include <vector>

namespace {

using Estimator =
    pcl::FPFHEstimation<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33>;

void
fillHistogram(Eigen::MatrixXf& histogram, int seed)
{
  for (Eigen::Index row = 0; row < histogram.rows(); ++row) {
    for (Eigen::Index bin = 0; bin < histogram.cols(); ++bin) {
      histogram(row, bin) =
          0.125f * static_cast<float>(1 + ((row + seed) * (bin + seed + 1)) % 31);
    }
  }
}

Eigen::VectorXf
weightScalar(const std::array<Eigen::MatrixXf, 3>& histograms,
             const pcl::Indices& indices,
             const std::vector<float>& distances)
{
  const std::array<Eigen::Index, 3> offsets = {
      0, histograms[0].cols(), histograms[0].cols() + histograms[1].cols()};
  Eigen::VectorXf result(offsets[2] + histograms[2].cols());
  result.setZero();
  std::array<double, 3> sums{};

  for (std::size_t neighbor = 0; neighbor < indices.size(); ++neighbor) {
    if (distances[neighbor] == 0.0f)
      continue;

    const float weight = 1.0f / distances[neighbor];
    for (std::size_t segment = 0; segment < histograms.size(); ++segment) {
      for (Eigen::Index bin = 0; bin < histograms[segment].cols(); ++bin) {
        const float value = histograms[segment](indices[neighbor], bin) * weight;
        sums[segment] += value;
        result[offsets[segment] + bin] += value;
      }
    }
  }

  for (std::size_t segment = 0; segment < histograms.size(); ++segment) {
    const float scale =
        sums[segment] == 0.0 ? 0.0f : static_cast<float>(100.0 / sums[segment]);
    result.segment(offsets[segment], histograms[segment].cols()) *= scale;
  }
  return result;
}

void
expectVectorNear(const Eigen::VectorXf& actual,
                 const Eigen::VectorXf& expected,
                 float tolerance)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (Eigen::Index bin = 0; bin < actual.size(); ++bin)
    EXPECT_NEAR(actual[bin], expected[bin], tolerance) << "bin=" << bin;
}

TEST(FPFHWeighting, HandlesRemappedRowsAndZeroDistance)
{
  std::array<Eigen::MatrixXf, 3> histograms = {
      Eigen::MatrixXf(17, 11), Eigen::MatrixXf(17, 11), Eigen::MatrixXf(17, 11)};
  fillHistogram(histograms[0], 1);
  fillHistogram(histograms[1], 3);
  fillHistogram(histograms[2], 7);

  const pcl::Indices indices = {0, 12, 4, 15, 2, 9, 6};
  const std::vector<float> distances = {0.0f, 0.39f, 0.48f, 0.57f, 0.66f, 0.75f, 0.84f};
  const Eigen::VectorXf expected = weightScalar(histograms, indices, distances);

  Eigen::VectorXf actual;
  Estimator estimator;
  estimator.weightPointSPFHSignature(
      histograms[0], histograms[1], histograms[2], indices, distances, actual);

  expectVectorNear(actual, expected, 2e-4f);
}

TEST(FPFHWeighting, SupportsNonDefaultBinCounts)
{
  std::array<Eigen::MatrixXf, 3> histograms = {
      Eigen::MatrixXf(8, 7), Eigen::MatrixXf(8, 5), Eigen::MatrixXf(8, 9)};
  fillHistogram(histograms[0], 2);
  fillHistogram(histograms[1], 5);
  fillHistogram(histograms[2], 8);

  const pcl::Indices indices = {7, 1, 5, 2};
  const std::vector<float> distances = {0.25f, 0.5f, 0.75f, 1.0f};
  const Eigen::VectorXf expected = weightScalar(histograms, indices, distances);

  Eigen::VectorXf actual;
  Estimator estimator;
  estimator.weightPointSPFHSignature(
      histograms[0], histograms[1], histograms[2], indices, distances, actual);

  expectVectorNear(actual, expected, 1e-5f);
}

} // namespace
