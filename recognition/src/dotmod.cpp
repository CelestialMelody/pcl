/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 */

#include <pcl/recognition/dotmod.h>

#if defined(__RVV10__)
#include <riscv_vector.h>
#endif

#include <algorithm>
#include <fstream>

#if defined(__RVV10__)
namespace {

// Scores one template window without materializing it. QuantizedMap stores entry
// (x, y) at offset y * width + x, so the window that QuantizedMap::getSubMap ()
// would have copied holds exactly the bytes this strided read visits.
std::size_t
dotmodScoreWindowDirectRVV(const unsigned char* image_data,
                           const std::size_t image_width,
                           const std::size_t window_x,
                           const std::size_t window_y,
                           const std::size_t window_width,
                           const std::size_t window_height,
                           const unsigned char* template_data)
{
  std::size_t score = 0;
  for (std::size_t row_index = 0; row_index < window_height; ++row_index) {
    const unsigned char* image_row =
        image_data + (window_y + row_index) * image_width + window_x;
    const unsigned char* template_row = template_data + row_index * window_width;

    std::size_t col_index = 0;
    while (col_index < window_width) {
      const std::size_t vl = __riscv_vsetvl_e8m8(window_width - col_index);
      const vuint8m8_t image_values = __riscv_vle8_v_u8m8(image_row + col_index, vl);
      const vuint8m8_t template_values =
          __riscv_vle8_v_u8m8(template_row + col_index, vl);
      const vuint8m8_t hits = __riscv_vand_vv_u8m8(image_values, template_values, vl);
      const vbool1_t hit_mask = __riscv_vmsne_vx_u8m8_b1(hits, 0, vl);
      score += __riscv_vcpop_m_b1(hit_mask, vl);
      col_index += vl;
    }
  }
  return score;
}

} // namespace
#endif // defined(__RVV10__)

//////////////////////////////////////////////////////////////////////////////////////////////
pcl::DOTMOD::
DOTMOD(std::size_t template_width,
       std::size_t template_height) :
  template_width_ (template_width),
  template_height_ (template_height)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////
pcl::DOTMOD::
~DOTMOD() = default;

//////////////////////////////////////////////////////////////////////////////////////////////
size_t 
pcl::DOTMOD::
createAndAddTemplate (const std::vector<pcl::DOTModality*> & modalities,
                      const std::vector<pcl::MaskMap*> & masks,
                      std::size_t template_anker_x,
                      std::size_t template_anker_y,
                      const pcl::RegionXY & region)
{
  DenseQuantizedMultiModTemplate dotmod_template;

  RegionXY actual_template_region;
  actual_template_region.x = static_cast<int> (template_anker_x);
  actual_template_region.y = static_cast<int> (template_anker_y);
  actual_template_region.width = static_cast<int> (template_width_);
  actual_template_region.height = static_cast<int> (template_height_);

  // get template data from available modalities
  const std::size_t nr_modalities = modalities.size();
  dotmod_template.modalities.resize (nr_modalities);
  for (std::size_t modality_index = 0; modality_index < nr_modalities; ++modality_index)
  {
    const MaskMap & mask = *(masks[modality_index]);
    const QuantizedMap & data = modalities[modality_index]->computeInvariantQuantizedMap (mask, actual_template_region);

    const std::size_t width = data.getWidth ();
    const std::size_t height = data.getHeight ();

    dotmod_template.modalities[modality_index].features.resize (width*height);

    for (std::size_t row_index = 0; row_index < height; ++row_index)
    {
      for (std::size_t col_index = 0; col_index < width; ++col_index)
      {
        dotmod_template.modalities[modality_index].features[row_index*width + col_index] = data (col_index, row_index);
      }
    }
  }

  // set region relative to the center
  dotmod_template.region.x = region.x - static_cast<int> (template_anker_x);
  dotmod_template.region.y = region.y - static_cast<int> (template_anker_y);
  dotmod_template.region.width = region.width;
  dotmod_template.region.height = region.height;

  // add template to template storage
  templates_.push_back(dotmod_template);

  return templates_.size () - 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
detectTemplates (const std::vector<DOTModality*> & modalities, 
                 const float template_response_threshold,
                 std::vector<DOTMODDetection> & detections,
                 const std::size_t bin_size ) const
{
#if defined(__RVV10__)
  // The RVV implementation reads the dominant map directly instead of materializing a
  // per-window sub-map, so the two implementations are kept side by side and the public
  // entry point only dispatches. Both produce the same detections in the same order.
  detectTemplatesRVV (modalities, template_response_threshold, detections, bin_size);
#else
  detectTemplatesStandard (modalities, template_response_threshold, detections, bin_size);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
detectTemplatesStandard (const std::vector<DOTModality*> & modalities,
                         const float template_response_threshold,
                         std::vector<DOTMODDetection> & detections,
                         const std::size_t bin_size ) const
{
  //std::cerr << ">> detectTemplates (...)" << std::endl;

  std::vector<QuantizedMap> maps;
  const std::size_t nr_modalities = modalities.size ();

  //std::cerr << "nr_modalities: " << nr_modalities << std::endl;

  for (std::size_t modality_index = 0; modality_index < nr_modalities; ++modality_index)
  {
    QuantizedMap &map = modalities[modality_index]->getDominantQuantizedMap ();
    maps.push_back(map);
  }

  //std::cerr << "1" << std::endl;

  const std::size_t width = maps[0].getWidth ();
  const std::size_t height = maps[0].getHeight ();
  const std::size_t nr_templates = templates_.size ();

  const std::size_t nr_template_horizontal_bins = template_width_ / bin_size;
  const std::size_t nr_template_vertical_bins = template_height_ / bin_size;

  //std::cerr << "2" << std::endl;

  float best_response = 0.0f;
  std::vector<float> responses (nr_templates, 0.0f);
  for (std::size_t row_index = 0; row_index < (height - nr_template_vertical_bins); ++row_index)
  {
    for (std::size_t col_index = 0; col_index < (width - nr_template_horizontal_bins); ++col_index)
    {
      std::fill (responses.begin (), responses.end (), 0.0f);

      for (std::size_t modality_index = 0; modality_index < nr_modalities; ++modality_index)
      {
        const QuantizedMap map = maps[modality_index].getSubMap (col_index, row_index, nr_template_horizontal_bins, nr_template_vertical_bins);

        const unsigned char * image_data = map.getData ();
        for (std::size_t template_index = 0; template_index < nr_templates; ++template_index)
        {
          const unsigned char * template_data = templates_[template_index].modalities[modality_index].features.data();
          for (std::size_t data_index = 0; data_index < (nr_template_horizontal_bins*nr_template_vertical_bins); ++data_index)
          {
            if ((image_data[data_index] & template_data[data_index]) != 0)
              responses[template_index] += 1.0f;
          }
        }
      }

      // find templates with response over threshold
      const float scaling_factor = 1.0f / static_cast<float>(nr_template_horizontal_bins * nr_template_vertical_bins);
      for (std::size_t template_index = 0; template_index < nr_templates; ++template_index)
      {
        const float response = responses[template_index] * scaling_factor;

        if (response > template_response_threshold)
        {
          DOTMODDetection detection;
          detection.score = response;
          detection.template_id = template_index;
          detection.bin_x = col_index;
          detection.bin_y = row_index;

          detections.push_back (detection);
        }

        if (response > best_response)
          best_response = response;
      }
    }
  }

  //std::cerr << "best_response: " << best_response << std::endl;

  //std::cerr << "<< detectTemplates (...)" << std::endl;
}

#if defined(__RVV10__)
////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
detectTemplatesRVV (const std::vector<DOTModality*> & modalities,
                    const float template_response_threshold,
                    std::vector<DOTMODDetection> & detections,
                    const std::size_t bin_size ) const
{
  std::vector<QuantizedMap> maps;
  const std::size_t nr_modalities = modalities.size ();

  for (std::size_t modality_index = 0; modality_index < nr_modalities; ++modality_index)
  {
    QuantizedMap &map = modalities[modality_index]->getDominantQuantizedMap ();
    maps.push_back(map);
  }

  const std::size_t width = maps[0].getWidth ();
  const std::size_t height = maps[0].getHeight ();
  const std::size_t nr_templates = templates_.size ();

  const std::size_t nr_template_horizontal_bins = template_width_ / bin_size;
  const std::size_t nr_template_vertical_bins = template_height_ / bin_size;

  float best_response = 0.0f;
  std::vector<float> responses (nr_templates, 0.0f);
  for (std::size_t row_index = 0; row_index < (height - nr_template_vertical_bins); ++row_index)
  {
    for (std::size_t col_index = 0; col_index < (width - nr_template_horizontal_bins); ++col_index)
    {
      std::fill (responses.begin (), responses.end (), 0.0f);

      for (std::size_t modality_index = 0; modality_index < nr_modalities; ++modality_index)
      {
        const unsigned char * image_data = maps[modality_index].getData ();
        const std::size_t image_width = maps[modality_index].getWidth ();

        for (std::size_t template_index = 0; template_index < nr_templates; ++template_index)
        {
          const unsigned char * template_data = templates_[template_index].modalities[modality_index].features.data();
          responses[template_index] +=
              static_cast<float>(dotmodScoreWindowDirectRVV(image_data,
                                                            image_width,
                                                            col_index,
                                                            row_index,
                                                            nr_template_horizontal_bins,
                                                            nr_template_vertical_bins,
                                                            template_data));
        }
      }

      // find templates with response over threshold
      const float scaling_factor = 1.0f / static_cast<float>(nr_template_horizontal_bins * nr_template_vertical_bins);
      for (std::size_t template_index = 0; template_index < nr_templates; ++template_index)
      {
        const float response = responses[template_index] * scaling_factor;

        if (response > template_response_threshold)
        {
          DOTMODDetection detection;
          detection.score = response;
          detection.template_id = template_index;
          detection.bin_x = col_index;
          detection.bin_y = row_index;

          detections.push_back (detection);
        }

        if (response > best_response)
          best_response = response;
      }
    }
  }
}
#endif // defined(__RVV10__)

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
saveTemplates (const char * file_name) const
{
  std::ofstream file_stream;
  file_stream.open (file_name, std::ofstream::out | std::ofstream::binary);

  serialize (file_stream);

  file_stream.close ();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
loadTemplates (const char * file_name)
{
  std::ifstream file_stream;
  file_stream.open (file_name, std::ofstream::in | std::ofstream::binary);

  deserialize (file_stream);

  file_stream.close ();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
serialize (std::ostream & stream) const
{
  const int nr_templates = static_cast<int> (templates_.size ());
  write (stream, nr_templates);
  for (int template_index = 0; template_index < nr_templates; ++template_index)
    templates_[template_index].serialize (stream);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::DOTMOD::
deserialize (std::istream & stream)
{
  templates_.clear ();

  int nr_templates;
  read (stream, nr_templates);
  templates_.resize (nr_templates);
  for (int template_index = 0; template_index < nr_templates; ++template_index)
    templates_[template_index].deserialize (stream);
}
