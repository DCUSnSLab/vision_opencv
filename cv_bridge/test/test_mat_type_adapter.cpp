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

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "cv_bridge/cv_mat_sensor_msgs_image_type_adapter.hpp"

using Adapter = rclcpp::TypeAdapter<cv_bridge::ROSCvMatContainer, sensor_msgs::msg::Image>;

namespace
{
std::unique_ptr<sensor_msgs::msg::Image> image(uint8_t value)
{
  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->height = 4;
  msg->width = 6;
  msg->step = 6;
  msg->encoding = "mono8";
  msg->data.assign(24, value);
  return msg;
}
}  // namespace

TEST(MatTypeAdapter, ConvertToCustomOwnsOneCopy)
{
  auto msg = image(3);
  cv_bridge::ROSCvMatContainer container;
  Adapter::convert_to_custom(*msg, container);
  EXPECT_NE(msg->data.data(), container.cv_mat().data);
  EXPECT_EQ(3, container.cv_mat().at<uint8_t>(3, 5));
  msg->data[0] = 9;
  EXPECT_EQ(3, container.cv_mat().at<uint8_t>(0, 0));
}

TEST(MatTypeAdapter, MoveKeepsTheImage)
{
  auto msg = image(4);
  const uint8_t * data = msg->data.data();
  cv_bridge::ROSCvMatContainer first(std::move(msg));
  cv_bridge::ROSCvMatContainer second(std::move(first));
  EXPECT_EQ(data, second.cv_mat().data);
  cv_bridge::ROSCvMatContainer third;
  third = std::move(second);
  EXPECT_EQ(data, third.cv_mat().data);
}

TEST(MatTypeAdapter, CopyOfOwnedMessageIsIndependent)
{
  cv_bridge::ROSCvMatContainer original(image(5));
  cv_bridge::ROSCvMatContainer copy(original);
  EXPECT_NE(original.cv_mat().data, copy.cv_mat().data);
  copy.cv_mat().at<uint8_t>(1, 1) = 7;
  EXPECT_EQ(5, original.cv_mat().at<uint8_t>(1, 1));
  EXPECT_EQ(7, copy.cv_mat().at<uint8_t>(1, 1));
  sensor_msgs::msg::Image out;
  Adapter::convert_to_ros_message(copy, out);
  EXPECT_EQ(7, out.data[1 * 6 + 1]);
  EXPECT_EQ(5, out.data[0]);
}

TEST(MatTypeAdapter, CopyKeepsEncodingOverride)
{
  cv::Mat depth(2, 3, CV_16UC1, cv::Scalar(1000));
  cv_bridge::ROSCvMatContainer original(depth, std_msgs::msg::Header(), false, "16UC1");
  cv_bridge::ROSCvMatContainer copy(original);
  cv_bridge::ROSCvMatContainer assigned;
  assigned = original;
  sensor_msgs::msg::Image out;
  Adapter::convert_to_ros_message(copy, out);
  EXPECT_EQ("16UC1", out.encoding);
  Adapter::convert_to_ros_message(assigned, out);
  EXPECT_EQ("16UC1", out.encoding);
}

TEST(MatTypeAdapter, SharedConversionReferencesTheMessage)
{
  std::shared_ptr<const sensor_msgs::msg::Image> msg = image(6);
  auto container = Adapter::convert_to_custom(msg);
  EXPECT_EQ(msg->data.data(), container->cv_mat().data);
  EXPECT_EQ(msg, container->get_sensor_msgs_msg_image_pointer());
  EXPECT_EQ(6, container->cv_mat().at<uint8_t>(2, 2));
}
