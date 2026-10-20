// Copyright 2026 nevlife
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/mat.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

// visibility_control.h must come before cv_bridge_export.h, which only defines what is missing.
#include "cv_bridge/cv_gpu_mat_sensor_msgs_image_type_adapter.hpp"
#include "cv_bridge/cv_bridge.hpp"

namespace cv_bridge
{

namespace
{
// Same table as ROSCvMatContainer, so both containers publish the same encodings.
std::string
mat_type2encoding(int mat_type)
{
  switch (mat_type) {
    case CV_8UC1:
      return "mono8";
    case CV_8UC3:
      return "bgr8";
    case CV_16SC1:
      return "mono16";
    case CV_8UC4:
      return "rgba8";
    default:
      throw std::runtime_error("unsupported encoding type");
  }
}
}  // namespace

ROSCvGpuMatContainer::ROSCvGpuMatContainer(const ROSCvGpuMatContainer & other)
: header_(other.header_),
  frame_(other.frame_.clone()),
  is_bigendian_(other.is_bigendian_),
  encoding_override_(other.encoding_override_)
{}

ROSCvGpuMatContainer &
ROSCvGpuMatContainer::operator=(const ROSCvGpuMatContainer & other)
{
  if (this != &other) {
    header_ = other.header_;
    frame_ = other.frame_.clone();
    is_bigendian_ = other.is_bigendian_;
    encoding_override_ = other.encoding_override_;
  }
  return *this;
}

ROSCvGpuMatContainer::ROSCvGpuMatContainer(
  const cv::cuda::GpuMat & gpu_mat_frame,
  const std_msgs::msg::Header & header,
  bool is_bigendian,
  std::optional<std::string> encoding_override)
: header_(header),
  frame_(gpu_mat_frame),
  is_bigendian_(is_bigendian),
  encoding_override_(std::move(encoding_override))
{}

ROSCvGpuMatContainer::ROSCvGpuMatContainer(const sensor_msgs::msg::Image & sensor_msgs_image)
: header_(sensor_msgs_image.header),
  is_bigendian_(sensor_msgs_image.is_bigendian),
  encoding_override_(sensor_msgs_image.encoding)
{
  const cv::Mat host(
    static_cast<int>(sensor_msgs_image.height),
    static_cast<int>(sensor_msgs_image.width),
    getCvType(sensor_msgs_image.encoding),
    const_cast<uint8_t *>(sensor_msgs_image.data.data()),
    sensor_msgs_image.step);
  frame_.upload(host);
}

const cv::cuda::GpuMat &
ROSCvGpuMatContainer::cv_gpu_mat() const
{
  return frame_;
}

cv::cuda::GpuMat
ROSCvGpuMatContainer::cv_gpu_mat()
{
  return frame_;
}

const std_msgs::msg::Header &
ROSCvGpuMatContainer::header() const
{
  return header_;
}

std_msgs::msg::Header &
ROSCvGpuMatContainer::header()
{
  return header_;
}

void
ROSCvGpuMatContainer::get_sensor_msgs_msg_image_copy(
  sensor_msgs::msg::Image & sensor_msgs_image) const
{
  if (encoding_override_.has_value() && !encoding_override_.value().empty()) {
    sensor_msgs_image.encoding = encoding_override_.value();
  } else {
    sensor_msgs_image.encoding = mat_type2encoding(frame_.type());
  }
  sensor_msgs_image.header = header_;
  sensor_msgs_image.height = frame_.rows;
  sensor_msgs_image.width = frame_.cols;
  sensor_msgs_image.is_bigendian = is_bigendian_;
  // GpuMat rows are pitched; the message gets packed rows.
  const size_t step = static_cast<size_t>(frame_.cols) * frame_.elemSize();
  sensor_msgs_image.step = static_cast<sensor_msgs::msg::Image::_step_type>(step);
  sensor_msgs_image.data.resize(step * frame_.rows);
  cv::Mat host(frame_.rows, frame_.cols, frame_.type(), sensor_msgs_image.data.data(), step);
  frame_.download(host);
}

bool
ROSCvGpuMatContainer::is_bigendian() const
{
  return is_bigendian_;
}

std::optional<std::string>
ROSCvGpuMatContainer::encoding_override() const
{
  return encoding_override_;
}

}  // namespace cv_bridge
