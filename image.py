import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import mediapy as media

model = hub.load("https://tfhub.dev/google/film/1")

def _pad_to_align(x, align):
    height, width = x.shape[-3:-1]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    bbox_to_pad = {
        'offset_height': height_to_pad // 2,
        'offset_width': width_to_pad // 2,
        'target_height': height + height_to_pad,
        'target_width': width + width_to_pad
    }
    padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)

    bbox_to_crop = {
        'offset_height': height_to_pad // 2,
        'offset_width': width_to_pad // 2,
        'target_height': height,
        'target_width': width
    }
    return padded_x, bbox_to_crop

class Interpolator:
    def __init__(self, align: int = 64) -> None:
        self._model = hub.load("https://tfhub.dev/google/film/1")
        self._align = align

    def __call__(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> np.ndarray:
        if self._align is not None:
            x0, bbox_to_crop = _pad_to_align(x0, self._align)
            x1, _ = _pad_to_align(x1, self._align)

        inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
        result = self._model(inputs, training=False)
        image = result['image']

        if self._align is not None:
            image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image.numpy()

def load_local_image(image_path: str) -> np.ndarray:
    image_data = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / 255.0

image1 = load_local_image("frame1.jpeg")
image2 = load_local_image("frame2.jpeg")

interpolator = Interpolator()

time_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
frames = []

for time_pos in time_positions:
    if time_pos == 0.0:
        frames.append(image1)
    elif time_pos == 1.0:
        frames.append(image2)
    else:
        time_array = np.array([time_pos], dtype=np.float32)
        interpolated_frame = interpolator(
            np.expand_dims(image1, axis=0),
            np.expand_dims(image2, axis=0),
            time_array
        )[0]
        frames.append(interpolated_frame)

media.show_video(frames, fps=5)
media.write_video('result.mp4', frames, fps=5)
