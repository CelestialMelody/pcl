/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2026-, Open Perception Inc.
 *
 *  All rights reserved
 */

#include <pcl/recognition/dotmod.h>
#include <pcl/test/gtest.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace {

// A modality with fixed contents, so that detection can be checked against a
// hand-computed response. Quantized maps are row-major: entry (x, y) is stored
// at offset y * width + x.
class FixedDOTModality final : public pcl::DOTModality {
public:
  FixedDOTModality(const pcl::QuantizedMap& image, const pcl::QuantizedMap& templ)
  : image_(image), templ_(templ)
  {}

  pcl::QuantizedMap&
  getDominantQuantizedMap() override
  {
    return image_;
  }

  pcl::QuantizedMap
  computeInvariantQuantizedMap(const pcl::MaskMap&, const pcl::RegionXY&) override
  {
    return templ_;
  }

private:
  pcl::QuantizedMap image_;
  pcl::QuantizedMap templ_;
};

// Drives the public template and detection entry points with one modality and one
// template. The template map must hold exactly one entry per scanned window cell,
// that is template_width / bin_size by template_height / bin_size entries.
std::vector<pcl::DOTMODDetection>
detect(const pcl::QuantizedMap& image,
       const pcl::QuantizedMap& templ,
       std::size_t template_width,
       std::size_t template_height,
       float threshold)
{
  constexpr std::size_t bin_size = 2;

  FixedDOTModality modality(image, templ);
  std::vector<pcl::DOTModality*> modalities{&modality};

  pcl::MaskMap mask(image.getWidth(), image.getHeight());
  std::vector<pcl::MaskMap*> masks{&mask};

  pcl::RegionXY region;
  region.x = 0;
  region.y = 0;
  region.width = static_cast<int>(template_width);
  region.height = static_cast<int>(template_height);

  pcl::DOTMOD dotmod(template_width, template_height);
  dotmod.createAndAddTemplate(modalities, masks, 0, 0, region);

  std::vector<pcl::DOTMODDetection> detections;
  dotmod.detectTemplates(modalities, threshold, detections, bin_size);
  return detections;
}

// A 2x2 template whose four bytes carry four distinct bits.
pcl::QuantizedMap
makeTemplate()
{
  pcl::QuantizedMap templ(2, 2);
  templ(0, 0) = 1;
  templ(1, 0) = 2;
  templ(0, 1) = 4;
  templ(1, 1) = 8;
  return templ;
}

// A 3x3 image whose top-left 2x2 window reproduces the template exactly.
pcl::QuantizedMap
makeImage()
{
  pcl::QuantizedMap image(3, 3);
  const unsigned char values[] = {1, 2, 16, 4, 8, 32, 64, 128, 1};
  std::copy(std::begin(values), std::end(values), image.getData());
  return image;
}

} // namespace

TEST(DOTMOD, DetectsExactTemplate)
{
  const auto detections = detect(makeImage(), makeTemplate(), 4, 4, 0.99f);

  ASSERT_EQ(detections.size(), 1u);
  EXPECT_EQ(detections[0].bin_x, 0u);
  EXPECT_EQ(detections[0].bin_y, 0u);
  EXPECT_EQ(detections[0].template_id, 0u);
  EXPECT_FLOAT_EQ(detections[0].score, 1.0f);
}

TEST(DOTMOD, RequiresResponseAboveThreshold)
{
  const auto detections = detect(makeImage(), makeTemplate(), 4, 4, 1.0f);

  EXPECT_TRUE(detections.empty());
}

// A 5x4 image scanned with a 2x2 window yields six candidate positions. The
// matching bytes are placed at window origin (2, 1), so a wrong window origin
// would score a different set of bytes and lose the detection.
TEST(DOTMOD, DetectsTemplateAtNonZeroWindowOffset)
{
  pcl::QuantizedMap image(5, 4);
  image(2, 1) = 1;
  image(3, 1) = 2;
  image(2, 2) = 4;
  image(3, 2) = 8;

  const auto detections = detect(image, makeTemplate(), 4, 4, 0.99f);

  ASSERT_EQ(detections.size(), 1u);
  EXPECT_EQ(detections[0].bin_x, 2u);
  EXPECT_EQ(detections[0].bin_y, 1u);
  EXPECT_FLOAT_EQ(detections[0].score, 1.0f);
}

// A 600 entry wide window is longer than the vector registers used by the
// optimized scoring path, so the final partial iteration must still count every
// remaining byte. A full match only reaches score 1.0 when it does.
TEST(DOTMOD, CountsEveryByteOfWideWindow)
{
  constexpr std::size_t image_width = 601;
  constexpr std::size_t image_height = 2;
  constexpr std::size_t template_width = 1200;
  constexpr std::size_t template_height = 2;

  pcl::QuantizedMap image(image_width, image_height);
  std::fill(image.getData(), image.getData() + image_width * image_height, 1);

  pcl::QuantizedMap templ(template_width / 2, template_height / 2);
  std::fill(templ.getData(), templ.getData() + templ.getWidth() * templ.getHeight(), 1);

  const auto detections = detect(image, templ, template_width, template_height, 0.99f);

  ASSERT_EQ(detections.size(), 1u);
  EXPECT_EQ(detections[0].bin_x, 0u);
  EXPECT_EQ(detections[0].bin_y, 0u);
  EXPECT_FLOAT_EQ(detections[0].score, 1.0f);
}
