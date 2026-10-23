# Copyright 2026 nevlife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from cv_bridge.type_adapter import CV_IMAGE_ADAPTER, CvImage
import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def test_conversions_keep_pixels_header_and_encoding():
    pixels = np.random.randint(0, 255, (4, 5, 3), np.uint8)
    header = Header(frame_id='camera')
    header.stamp.sec = 7
    message = CV_IMAGE_ADAPTER.to_ros(CvImage(pixels, 'rgb8', header))
    assert (message.height, message.width, message.encoding) == (4, 5, 'rgb8')
    assert message.header.frame_id == 'camera' and message.header.stamp.sec == 7
    image = CV_IMAGE_ADAPTER.from_ros(message)
    np.testing.assert_array_equal(image.image, pixels)
    assert not image.image.flags.writeable
    assert image.encoding == 'rgb8' and image.header.frame_id == 'camera'
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert CV_IMAGE_ADAPTER.to_ros(CvImage(depth)).encoding == '32FC1'


@pytest.mark.parametrize('broker', [False, True], ids=['rmw', 'broker'])
def test_node_delivers_cv_images(broker, monkeypatch):
    monkeypatch.setenv('RCL_INTRA_PROCESS', '1' if broker else '0')
    broker = broker and os.environ.get('RMW_IMPLEMENTATION') == 'rmw_zenoh_cpp'
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('cv_image_adapter', context=context)
    executor = SingleThreadedExecutor(context=context)
    received, messages = [], []
    try:
        publisher = node.create_publisher(CvImage, 'cv_image', 10)
        node.create_subscription(CvImage, 'cv_image', received.append, 10)
        node.create_subscription(Image, 'cv_image', messages.append, 10)
        executor.add_node(node)
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() < 2:
            assert time.monotonic() < deadline
            executor.spin_once(timeout_sec=0.01)
        sent = CvImage(np.random.randint(0, 255, (48, 64, 3), np.uint8), 'bgr8',
                       Header(frame_id='camera'))
        publisher.publish(sent)
        while not received or not messages:
            assert time.monotonic() < deadline
            executor.spin_once(timeout_sec=0.01)
        assert (received[0] is sent) == broker
        np.testing.assert_array_equal(received[0].image, sent.image)
        assert received[0].encoding == 'bgr8' and received[0].header.frame_id == 'camera'
        assert messages[0].encoding == 'bgr8' and messages[0].step == 64 * 3
    finally:
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
