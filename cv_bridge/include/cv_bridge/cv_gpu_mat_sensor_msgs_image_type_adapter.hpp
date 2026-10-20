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

#ifndef CV_BRIDGE__CV_GPU_MAT_SENSOR_MSGS_IMAGE_TYPE_ADAPTER_HPP_
#define CV_BRIDGE__CV_GPU_MAT_SENSOR_MSGS_IMAGE_TYPE_ADAPTER_HPP_

#include <optional>
#include <string>

#include "opencv2/core/cuda.hpp"

#include "rclcpp/type_adapter.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "cv_bridge/visibility_control.h"

namespace cv_bridge
{

/// A container of a cv::cuda::GpuMat and ROS header, adapted to sensor_msgs::msg::Image.
/**
 * Intra-process subscribers that take this type receive the published GpuMat
 * without leaving device memory.
 * Inter-process subscribers and subscribers of sensor_msgs::msg::Image receive
 * a host copy made with cv::cuda::GpuMat::download().
 * A sensor_msgs::msg::Image received into this type is uploaded to the device.
 *
 * Copying the container clones the device data, like ROSCvMatContainer clones
 * its cv::Mat.
 */
class ROSCvGpuMatContainer
{
public:
  CV_BRIDGE_PUBLIC
  ROSCvGpuMatContainer() = default;

  CV_BRIDGE_PUBLIC
  ROSCvGpuMatContainer(const ROSCvGpuMatContainer & other);

  CV_BRIDGE_PUBLIC
  ROSCvGpuMatContainer & operator=(const ROSCvGpuMatContainer & other);

  CV_BRIDGE_PUBLIC
  ROSCvGpuMatContainer(ROSCvGpuMatContainer && other) = default;

  CV_BRIDGE_PUBLIC
  ROSCvGpuMatContainer & operator=(ROSCvGpuMatContainer && other) = default;

  /// Shallow copy the given cv::cuda::GpuMat into this class.
  CV_BRIDGE_PUBLIC
  ROSCvGpuMatContainer(
    const cv::cuda::GpuMat & gpu_mat_frame,
    const std_msgs::msg::Header & header,
    bool is_bigendian = false,
    std::optional<std::string> encoding_override = std::nullopt);

  /// Upload the sensor_msgs::msg::Image to the device.
  CV_BRIDGE_PUBLIC
  explicit ROSCvGpuMatContainer(const sensor_msgs::msg::Image & sensor_msgs_image);

  /// Const access the cv::cuda::GpuMat in this class.
  CV_BRIDGE_PUBLIC
  const cv::cuda::GpuMat &
  cv_gpu_mat() const;

  /// Get a shallow copy of the cv::cuda::GpuMat that is in this class.
  CV_BRIDGE_PUBLIC
  cv::cuda::GpuMat
  cv_gpu_mat();

  CV_BRIDGE_PUBLIC
  const std_msgs::msg::Header &
  header() const;

  CV_BRIDGE_PUBLIC
  std_msgs::msg::Header &
  header();

  /// Download the image into a sensor_msgs::msg::Image with a packed row step.
  CV_BRIDGE_PUBLIC
  void
  get_sensor_msgs_msg_image_copy(sensor_msgs::msg::Image & sensor_msgs_image) const;

  CV_BRIDGE_PUBLIC
  bool
  is_bigendian() const;

  CV_BRIDGE_PUBLIC
  std::optional<std::string>
  encoding_override() const;

private:
  std_msgs::msg::Header header_;
  cv::cuda::GpuMat frame_;
  bool is_bigendian_ = false;
  std::optional<std::string> encoding_override_;
};

}  // namespace cv_bridge

template<>
struct rclcpp::TypeAdapter<cv_bridge::ROSCvGpuMatContainer, sensor_msgs::msg::Image>
{
  using is_specialized = std::true_type;
  using custom_type = cv_bridge::ROSCvGpuMatContainer;
  using ros_message_type = sensor_msgs::msg::Image;

  static
  void
  convert_to_ros_message(
    const custom_type & source,
    ros_message_type & destination)
  {
    source.get_sensor_msgs_msg_image_copy(destination);
  }

  static
  void
  convert_to_custom(
    const ros_message_type & source,
    custom_type & destination)
  {
    destination = cv_bridge::ROSCvGpuMatContainer(source);
  }
};

#endif  // CV_BRIDGE__CV_GPU_MAT_SENSOR_MSGS_IMAGE_TYPE_ADAPTER_HPP_
