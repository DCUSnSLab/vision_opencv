import cv_bridge
import numpy as np
import pytest
from sensor_msgs.msg import Image


def test_cvtColorForDisplay():
    # convert label image to display
    label = np.zeros((480, 640), dtype=np.int32)
    height, width = label.shape[:2]
    label_value = 0
    grid_num_y, grid_num_x = 3, 4
    for grid_row in range(grid_num_y):
        grid_size_y = height // grid_num_y
        min_y = grid_size_y * grid_row
        max_y = min_y + grid_size_y
        for grid_col in range(grid_num_x):
            grid_size_x = width // grid_num_x
            min_x = grid_size_x * grid_col
            max_x = min_x + grid_size_x
            label[min_y:max_y, min_x:max_x] = label_value
            label_value += 1
    label_viz = cv_bridge.cvtColorForDisplay(label, '32SC1', 'bgr8')
    assert label_viz.dtype == np.uint8
    assert label_viz.min() == 0
    assert label_viz.max() == 255

    # Check that mono8 conversion returns the right shape.
    bridge = cv_bridge.CvBridge()
    mono = np.random.random((100, 100)) * 255
    mono = mono.astype(np.uint8)

    input_msg = bridge.cv2_to_imgmsg(mono, encoding='mono8')
    output = bridge.imgmsg_to_cv2(input_msg, desired_encoding='mono8')
    assert output.shape == (100, 100)


@pytest.mark.parametrize('encoding,dtype,channels', [
    ('rgb8', 'u1', 3), ('bgr8', 'u1', 3), ('mono16', 'u2', 1), ('32FC1', 'f4', 1)])
@pytest.mark.parametrize('bigendian', [False, True])
@pytest.mark.parametrize('padding', [0, 1, 64])
def test_image_view(encoding, dtype, channels, bigendian, padding):
    bridge = cv_bridge.CvBridge()
    dtype = np.dtype(dtype).newbyteorder('>' if bigendian else '<')
    shape = (4, 5) + ((channels,) if channels > 1 else ())
    expected = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    step = 5 * dtype.itemsize * channels + padding
    message = Image(height=4, width=5, step=step, encoding=encoding,
                    is_bigendian=bigendian, data=bytes(4 * step))
    for row in range(4):
        payload = expected[row].tobytes()
        memoryview(message.data)[row * step:row * step + len(payload)] = payload
    original = bytes(message.data)
    view = bridge.imgmsg_to_cv2_view(message)
    np.testing.assert_array_equal(view, expected)
    assert view.dtype == dtype
    assert view.strides[0] == step
    assert np.shares_memory(view, np.asarray(message.data))
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.flat[0] = 1
    assert bytes(message.data) == original
    del message
    np.testing.assert_array_equal(view, expected)


@pytest.mark.parametrize('dtype,encoding', [('u1', 'rgb8'), ('>u2', 'mono16'), ('f4', '32FC1')])
def test_image_write_owned_strided(dtype, encoding):
    shape = (4, 6, 3) if encoding == 'rgb8' else (4, 6)
    image = np.arange(np.prod(shape)).reshape(shape).astype(dtype)[:, ::2]
    message = cv_bridge.CvBridge().cv2_to_imgmsg(image, encoding)
    assert bytes(message.data) == image.tobytes()
    assert message.step == len(message.data) // 4
    image[:] = 0
    assert any(message.data)


def test_image_view_invalid_step():
    message = Image(height=2, width=3, step=2, encoding='mono8', data=bytes(4))
    with pytest.raises(cv_bridge.CvBridgeError):
        cv_bridge.CvBridge().imgmsg_to_cv2_view(message)


def test_writable_image_view():
    message = Image(height=2, width=3, step=4, encoding='mono8', data=bytes(8))
    view = cv_bridge.CvBridge().imgmsg_to_cv2_view(message, readonly=False)
    view[1, 2] = 42
    assert message.data[6] == 42


@pytest.mark.parametrize('encoding,dtype,channels', [
    ('rgb8', 'u1', 3), ('bgr8', 'u1', 3), ('mono16', 'u2', 1), ('32FC1', 'f4', 1)])
def test_same_encoding_owns_output(encoding, dtype, channels):
    bridge = cv_bridge.CvBridge()
    shape = (4, 6) + ((channels,) if channels > 1 else ())
    image = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    message = bridge.cv2_to_imgmsg(image, encoding)
    converted = bridge.imgmsg_to_cv2(message, encoding)
    np.testing.assert_array_equal(converted, image)
    assert not np.shares_memory(converted, np.asarray(message.data))
    assert converted.flags.writeable
    converted[:] = 0
    assert bytes(message.data) == image.tobytes()


@pytest.mark.parametrize('encoding,channels', [('mono16', 1), ('bgr16', 3), ('16UC4', 4)])
@pytest.mark.parametrize('padding_pixels', [0, 2])
def test_bigendian_to_native(encoding, channels, padding_pixels):
    shape = (3, 5) + ((channels,) if channels > 1 else ())
    expected = (np.arange(np.prod(shape)) * 1000 + 3).reshape(shape).astype(np.uint16)
    step = (5 + padding_pixels) * 2 * channels
    message = Image(height=3, width=5, step=step, encoding=encoding,
                    is_bigendian=True, data=bytes(3 * step))
    for row in range(3):
        payload = expected[row].astype('>u2').tobytes()
        memoryview(message.data)[row * step:row * step + len(payload)] = payload
    image = cv_bridge.CvBridge().imgmsg_to_cv2(message)
    np.testing.assert_array_equal(image, expected)
    assert image.dtype.isnative


@pytest.mark.parametrize('desired_encoding', ['passthrough', 'bgr8', 'mono8'])
def test_compressed_desired_encoding(desired_encoding):
    import cv2
    bridge = cv_bridge.CvBridge()
    image = np.arange(4 * 6 * 3).reshape(4, 6, 3).astype(np.uint8)
    message = bridge.cv2_to_compressed_imgmsg(image, 'png')
    decoded = bridge.compressed_imgmsg_to_cv2(message, desired_encoding)
    if desired_encoding == 'mono8':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(decoded, image)
