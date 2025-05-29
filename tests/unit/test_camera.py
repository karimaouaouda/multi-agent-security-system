import unittest

from tensorflow.python.ops.ragged.ragged_check_ops import assert_type

from client.camera import Camera


class MyTestCase(unittest.TestCase):
    def test_camera_object_creation(self):
        camera = Camera()
        assert_type(camera, Camera)


if __name__ == '__main__':
    unittest.main()
