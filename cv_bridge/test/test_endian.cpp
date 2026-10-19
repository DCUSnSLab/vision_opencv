#include <boost/endian/conversion.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <gtest/gtest.h>
#include <memory>

TEST(CvBridgeTest, byteEncodingsShareWithEitherEndianFlag)
{
  for (const auto & encoding : {"rgb8", "bgr8", "mono8", "8UC3"}) {
    auto message = std::make_shared<sensor_msgs::msg::Image>();
    message->height = 2;
    message->width = 3;
    message->encoding = encoding;
    message->step = message->width * CV_ELEM_SIZE(cv_bridge::getCvType(encoding)) + 1;
    message->data.resize(message->height * message->step, 42);
    for (const bool bigendian : {false, true}) {
      message->is_bigendian = bigendian;
      auto view = cv_bridge::toCvShare(message, encoding);
      EXPECT_EQ(message->data.data(), view->image.data);
      EXPECT_EQ(message->step, view->image.step[0]);
      EXPECT_EQ(42, view->image.ptr(1)[0]);
    }
    auto view = cv_bridge::toCvShare(message, encoding);
    message.reset();
    EXPECT_EQ(42, view->image.ptr(1)[0]);
  }
}

TEST(CvBridgeTest, endianness)
{
  using namespace boost::endian;

  // Create an image of the type opposite to the platform
  sensor_msgs::msg::Image msg;
  msg.height = 1;
  msg.width = 1;
  msg.encoding = "32SC2";
  msg.step = 8;

  msg.data.resize(msg.step);
  int32_t * data = reinterpret_cast<int32_t *>(&msg.data[0]);

  // Write 1 and 2 in order, but with an endianness opposite to the platform
  if (order::native == order::little) {
    msg.is_bigendian = true;
    *(data++) = native_to_big(static_cast<int32_t>(1));
    *data = native_to_big(static_cast<int32_t>(2));
  } else {
    msg.is_bigendian = false;
    *(data++) = native_to_little(static_cast<int32_t>(1));
    *data = native_to_little(static_cast<int32_t>(2));
  }

  // Make sure the values are still the same
  cv_bridge::CvImageConstPtr img =
    cv_bridge::toCvShare(std::make_shared<sensor_msgs::msg::Image>(msg));
  EXPECT_EQ(img->image.at<cv::Vec2i>(0, 0)[0], 1);
  EXPECT_EQ(img->image.at<cv::Vec2i>(0, 0)[1], 2);
  // Make sure we cannot share data
  EXPECT_NE(img->image.data, &msg.data[0]);
}
