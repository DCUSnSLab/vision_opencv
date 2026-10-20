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

#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "cv_bridge/cv_gpu_mat_sensor_msgs_image_type_adapter.hpp"

using GpuImage = rclcpp::TypeAdapter<cv_bridge::ROSCvGpuMatContainer, sensor_msgs::msg::Image>;

namespace
{
bool has_cuda_device()
{
  return cv::cuda::getCudaEnabledDeviceCount() > 0;
}

std_msgs::msg::Header make_header()
{
  std_msgs::msg::Header header;
  header.frame_id = "camera";
  header.stamp.sec = 7;
  return header;
}

cv::Mat make_host_bgr(int rows, int cols)
{
  cv::Mat host(rows, cols, CV_8UC3);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols * 3; ++c) {
      host.ptr<uint8_t>(r)[c] = static_cast<uint8_t>(r * 31 + c);
    }
  }
  return host;
}
}  // namespace

class GpuMatTypeAdapter : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

// Needs no CUDA: an intra-process handoff must not touch the device data, so a GpuMat header
// over host memory is enough to see that the subscriber gets the published allocation.
TEST_F(GpuMatTypeAdapter, intra_process_passes_gpu_mat_without_conversion)
{
  auto node = std::make_shared<rclcpp::Node>(
    "gpu_mat_ipc", rclcpp::NodeOptions().use_intra_process_comms(true));
  std::vector<uint8_t> storage(4 * 512);
  const cv::cuda::GpuMat frame(4, 5, CV_8UC3, storage.data(), 512);

  std::unique_ptr<cv_bridge::ROSCvGpuMatContainer> received;
  auto sub = node->create_subscription<GpuImage>(
    "gpu_image", 1,
    [&received](std::unique_ptr<cv_bridge::ROSCvGpuMatContainer> msg) {
      received = std::move(msg);
    });
  auto pub = node->create_publisher<GpuImage>("gpu_image", 1);

  auto msg = std::make_unique<cv_bridge::ROSCvGpuMatContainer>(frame, make_header());
  const auto * published = msg.get();
  pub->publish(std::move(msg));

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!received && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_NE(received, nullptr);
  EXPECT_EQ(received.get(), published);
  EXPECT_EQ(received->cv_gpu_mat().data, storage.data());
  EXPECT_EQ(received->cv_gpu_mat().step, 512u);
  EXPECT_EQ(received->header().frame_id, "camera");
}

TEST_F(GpuMatTypeAdapter, conversion_packs_pitched_rows)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "OpenCV has no CUDA device support";
  }
  const cv::Mat host = make_host_bgr(4, 3);
  cv::cuda::GpuMat device;
  device.upload(host);
  ASSERT_GT(device.step, host.step);

  const cv_bridge::ROSCvGpuMatContainer container(device, make_header());
  sensor_msgs::msg::Image image;
  rclcpp::TypeAdapter<cv_bridge::ROSCvGpuMatContainer, sensor_msgs::msg::Image>::
  convert_to_ros_message(container, image);

  EXPECT_EQ(image.encoding, "bgr8");
  EXPECT_EQ(image.height, 4u);
  EXPECT_EQ(image.width, 3u);
  EXPECT_EQ(image.step, 9u);
  ASSERT_EQ(image.data.size(), 36u);
  EXPECT_EQ(image.header.frame_id, "camera");
  const cv::Mat view(4, 3, CV_8UC3, image.data.data(), image.step);
  EXPECT_EQ(cv::norm(view, host, cv::NORM_INF), 0.0);
}

TEST_F(GpuMatTypeAdapter, upload_keeps_encoding_and_padded_source_rows)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "OpenCV has no CUDA device support";
  }
  sensor_msgs::msg::Image image;
  image.header = make_header();
  image.encoding = "16UC1";
  image.height = 2;
  image.width = 3;
  image.step = 8;
  image.data = {1, 0, 2, 0, 3, 0, 0xee, 0xee, 4, 0, 5, 0, 6, 0, 0xee, 0xee};

  cv_bridge::ROSCvGpuMatContainer container;
  rclcpp::TypeAdapter<cv_bridge::ROSCvGpuMatContainer, sensor_msgs::msg::Image>::
  convert_to_custom(image, container);
  EXPECT_EQ(container.cv_gpu_mat().type(), CV_16UC1);

  sensor_msgs::msg::Image back;
  container.get_sensor_msgs_msg_image_copy(back);
  EXPECT_EQ(back.encoding, "16UC1");
  EXPECT_EQ(back.step, 6u);
  const std::vector<uint8_t> packed = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0};
  EXPECT_EQ(back.data, packed);
}

TEST_F(GpuMatTypeAdapter, copy_clones_and_published_frame_outlives_publisher)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "OpenCV has no CUDA device support";
  }
  cv::cuda::GpuMat device;
  device.upload(make_host_bgr(4, 3));
  auto original = std::make_unique<cv_bridge::ROSCvGpuMatContainer>(device, make_header());
  const cv_bridge::ROSCvGpuMatContainer copy(*original);
  EXPECT_NE(copy.cv_gpu_mat().data, device.data);

  device.release();
  original.reset();
  cv::Mat host;
  copy.cv_gpu_mat().download(host);
  EXPECT_EQ(cv::norm(host, make_host_bgr(4, 3), cv::NORM_INF), 0.0);
}

TEST_F(GpuMatTypeAdapter, image_subscriber_without_intra_process_receives_download)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "OpenCV has no CUDA device support";
  }
  auto pub_node = std::make_shared<rclcpp::Node>(
    "gpu_mat_pub", rclcpp::NodeOptions().use_intra_process_comms(true));
  auto sub_node = std::make_shared<rclcpp::Node>("gpu_mat_sub");
  sensor_msgs::msg::Image::SharedPtr received;
  auto sub = sub_node->create_subscription<sensor_msgs::msg::Image>(
    "gpu_image_inter", rclcpp::QoS(1).reliable(),
    [&received](sensor_msgs::msg::Image::SharedPtr msg) {received = msg;});
  auto pub = pub_node->create_publisher<GpuImage>("gpu_image_inter", rclcpp::QoS(1).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(pub_node);
  executor.add_node(sub_node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (pub->get_subscription_count() == 0 && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(10));
  }
  cv::cuda::GpuMat device;
  device.upload(make_host_bgr(4, 3));
  pub->publish(std::make_unique<cv_bridge::ROSCvGpuMatContainer>(device, make_header()));
  while (!received && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(10));
  }
  ASSERT_NE(received, nullptr);
  EXPECT_EQ(received->step, 9u);
  const cv::Mat view(4, 3, CV_8UC3, received->data.data(), received->step);
  EXPECT_EQ(cv::norm(view, make_host_bgr(4, 3), cv::NORM_INF), 0.0);
}
