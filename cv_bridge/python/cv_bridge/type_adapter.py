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

"""
An rclpy type adapter between OpenCV images and sensor_msgs/Image.

Importing this module registers ``CV_IMAGE_ADAPTER`` for :class:`CvImage`, so
``node.create_publisher(CvImage, ...)`` publishes images and
``node.create_subscription(CvImage, ...)`` delivers them, like the C++
``cv_bridge::ROSCvMatContainer`` adapter. In one process with the rcl broker, a
subscription receives the published :class:`CvImage` itself.
"""

from dataclasses import dataclass, field

import numpy
from rclpy.type_adapter import register_type_adapter, TypeAdapter
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .core import CvBridge


@dataclass
class CvImage:
    """
    An OpenCV image with the header and encoding of its message, like C++ ``cv_bridge::CvImage``.

    :param image: The pixels, as :meth:`CvBridge.cv2_to_imgmsg` accepts them.
    :param encoding: The image encoding, or ``'passthrough'`` for the OpenCV type of
        ``image``.
    :param header: The header of the message.
    """

    image: numpy.ndarray
    encoding: str = 'passthrough'
    header: Header = field(default_factory=Header)


_bridge = CvBridge()


def cv_image_to_imgmsg(value: CvImage) -> Image:
    """Convert a :class:`CvImage` to a sensor_msgs/Image."""
    return _bridge.cv2_to_imgmsg(value.image, value.encoding, value.header)


def imgmsg_to_cv_image(message: Image) -> CvImage:
    """
    Wrap a sensor_msgs/Image in a :class:`CvImage` without copying its pixels.

    ``image`` is a read-only view with the row step and byte order of the message, see
    :meth:`CvBridge.imgmsg_to_cv2_view`. With ``RCLPY_MESSAGE_ARRAYS=numpy`` it views the
    received message without any copy.
    """
    return CvImage(_bridge.imgmsg_to_cv2_view(message), message.encoding, message.header)


CV_IMAGE_ADAPTER = TypeAdapter(CvImage, Image, cv_image_to_imgmsg, imgmsg_to_cv_image)
register_type_adapter(CV_IMAGE_ADAPTER)
