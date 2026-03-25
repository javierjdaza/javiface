import os
import math
import base64
import warnings
from pathlib import Path
from typing import Union, Any, Optional, Dict, Tuple, List

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import cv2
import requests
import tensorflow as tf
from PIL import Image

warnings.filterwarnings("ignore")

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
tf_minor = int(tf.__version__.split(".", maxsplit=-1)[1])

if tf_version == 1 or (tf_version == 2 and tf_minor < 16):
    pass
else:
    try:
        import importlib.util
        if importlib.util.find_spec("tf_keras") is None:
            raise ImportError("tf_keras not found")
    except ImportError as err:
        raise ValueError(
            f"You have tensorflow {tf.__version__} and this requires "
            "tf-keras package. Please run `pip install tf-keras` "
            "or downgrade your tensorflow."
        ) from err

if tf_version == 2:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, BatchNormalization, ZeroPadding2D, Conv2D,
        ReLU, MaxPool2D, Add, UpSampling2D, concatenate, Softmax,
    )
else:
    from keras.models import Model
    from keras.layers import (
        Input, BatchNormalization, ZeroPadding2D, Conv2D,
        ReLU, MaxPool2D, Add, UpSampling2D, concatenate, Softmax,
    )


# ── Preprocess ────────────────────────────────────────────────────────────────

def get_image(img_uri: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img_uri, np.ndarray):
        img = img_uri.copy()
    elif isinstance(img_uri, str) and img_uri.startswith("data:image/"):
        encoded_data = img_uri.split(",")[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(img_uri, str) and img_uri.startswith("http"):
        response = requests.get(img_uri, stream=True, timeout=60)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    elif isinstance(img_uri, (str, Path)):
        if isinstance(img_uri, Path):
            img_uri = str(img_uri)
        if not os.path.isfile(img_uri):
            raise ValueError(f"Input image file path ({img_uri}) does not exist.")
        img = cv2.imread(img_uri)
    else:
        raise ValueError(f"Invalid image input - {img_uri}.")

    if len(img.shape) != 3 or np.prod(img.shape) == 0:
        raise ValueError("Input image needs to have 3 channels and must not be empty.")

    return img


def _scale_image(img: np.ndarray, scales: list, allow_upscaling: bool) -> tuple:
    img_h, img_w = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return img, im_scale


def preprocess_image(img: np.ndarray, allow_upscaling: bool) -> tuple:
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [1024, 1980]

    img, im_scale = _scale_image(img, scales, allow_upscaling)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]

    return im_tensor, img.shape[0:2], im_scale


# ── Postprocess ───────────────────────────────────────────────────────────────

def _find_euclidean_distance(source, test) -> float:
    if isinstance(source, list):
        source = np.array(source)
    if isinstance(test, list):
        test = np.array(test)
    diff = source - test
    return np.sqrt(np.sum(np.multiply(diff, diff)))


def alignment_procedure(img: np.ndarray, left_eye: tuple, right_eye: tuple, nose: tuple):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1

    a = _find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = _find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = _find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    if b != 0 and c != 0:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        cos_a = min(1.0, max(-1.0, cos_a))
        angle = (np.arccos(cos_a) * 180) / math.pi

        if direction == -1:
            angle = 90 - angle

        img = np.array(Image.fromarray(img).rotate(direction * angle))
    else:
        angle = 0.0

    return img, angle, direction


def rotate_facial_area(facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]):
    angle = angle * np.pi / 180
    height, weight = size

    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    x1 = max(int(x_new - (facial_area[2] - facial_area[0]) / 2), 0)
    y1 = max(int(y_new - (facial_area[3] - facial_area[1]) / 2), 0)
    x2 = min(int(x_new + (facial_area[2] - facial_area[0]) / 2), weight)
    y2 = min(int(y_new + (facial_area[3] - facial_area[1]) / 2), height)

    return (x1, y1, x2, y2)


def _pad_to_target(img: np.ndarray, target_size: Tuple[int, int], min_max_norm: bool = True) -> np.ndarray:
    factor = min(target_size[0] / img.shape[0], target_size[1] / img.shape[1])
    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    img = np.pad(
        img,
        ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
        "constant",
    )

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    if min_max_norm and img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img


def bbox_pred(boxes, box_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes


def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))

    boxes = boxes.astype(float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y

    return pred


def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def anchors_plane(height, width, stride, base_anchors):
    A = base_anchors.shape[0]
    c_0_2 = np.tile(np.arange(0, width)[np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
    c_1_3 = np.tile(np.arange(0, height)[:, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
    all_anchors = np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(
        base_anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1)
    )
    return all_anchors


def cpu_nms(dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)
    keep = []

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1, iy1, ix2, iy2, iarea = x1[i], y1[i], x2[i], y2[i], areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            w = max(0.0, min(ix2, x2[j]) - max(ix1, x1[j]) + 1)
            h = max(0.0, min(iy2, y2[j]) - max(iy1, y1[j]) + 1)
            ovr = (w * h) / (iarea + areas[j] - w * h)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep


# ── Model Builder ─────────────────────────────────────────────────────────────

def _load_weights(model: Model, weights_path) -> Model:
    model.load_weights(weights_path)
    return model


def build_model(weights_path) -> Model:
    data = Input(dtype=tf.float32, shape=(None, None, 3), name="data")

    bn_data = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn_data", trainable=False)(data)
    conv0_pad = ZeroPadding2D(padding=tuple([3, 3]))(bn_data)
    conv0 = Conv2D(filters=64, kernel_size=(7, 7), name="conv0", strides=[2, 2], padding="VALID", use_bias=False)(conv0_pad)
    bn0 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn0", trainable=False)(conv0)
    relu0 = ReLU(name="relu0")(bn0)
    pooling0_pad = ZeroPadding2D(padding=tuple([1, 1]))(relu0)
    pooling0 = MaxPool2D((3, 3), (2, 2), padding="valid", name="pooling0")(pooling0_pad)

    stage1_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit1_bn1", trainable=False)(pooling0)
    stage1_unit1_relu1 = ReLU(name="stage1_unit1_relu1")(stage1_unit1_bn1)
    stage1_unit1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), name="stage1_unit1_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit1_relu1)
    stage1_unit1_sc = Conv2D(filters=256, kernel_size=(1, 1), name="stage1_unit1_sc", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit1_relu1)
    stage1_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit1_bn2", trainable=False)(stage1_unit1_conv1)
    stage1_unit1_relu2 = ReLU(name="stage1_unit1_relu2")(stage1_unit1_bn2)
    stage1_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit1_relu2)
    stage1_unit1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), name="stage1_unit1_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit1_conv2_pad)
    stage1_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit1_bn3", trainable=False)(stage1_unit1_conv2)
    stage1_unit1_relu3 = ReLU(name="stage1_unit1_relu3")(stage1_unit1_bn3)
    stage1_unit1_conv3 = Conv2D(filters=256, kernel_size=(1, 1), name="stage1_unit1_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit1_relu3)
    plus0_v1 = Add()([stage1_unit1_conv3, stage1_unit1_sc])

    stage1_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit2_bn1", trainable=False)(plus0_v1)
    stage1_unit2_relu1 = ReLU(name="stage1_unit2_relu1")(stage1_unit2_bn1)
    stage1_unit2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), name="stage1_unit2_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit2_relu1)
    stage1_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit2_bn2", trainable=False)(stage1_unit2_conv1)
    stage1_unit2_relu2 = ReLU(name="stage1_unit2_relu2")(stage1_unit2_bn2)
    stage1_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit2_relu2)
    stage1_unit2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), name="stage1_unit2_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit2_conv2_pad)
    stage1_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit2_bn3", trainable=False)(stage1_unit2_conv2)
    stage1_unit2_relu3 = ReLU(name="stage1_unit2_relu3")(stage1_unit2_bn3)
    stage1_unit2_conv3 = Conv2D(filters=256, kernel_size=(1, 1), name="stage1_unit2_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit2_relu3)
    plus1_v2 = Add()([stage1_unit2_conv3, plus0_v1])

    stage1_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit3_bn1", trainable=False)(plus1_v2)
    stage1_unit3_relu1 = ReLU(name="stage1_unit3_relu1")(stage1_unit3_bn1)
    stage1_unit3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), name="stage1_unit3_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit3_relu1)
    stage1_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit3_bn2", trainable=False)(stage1_unit3_conv1)
    stage1_unit3_relu2 = ReLU(name="stage1_unit3_relu2")(stage1_unit3_bn2)
    stage1_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit3_relu2)
    stage1_unit3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), name="stage1_unit3_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit3_conv2_pad)
    stage1_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage1_unit3_bn3", trainable=False)(stage1_unit3_conv2)
    stage1_unit3_relu3 = ReLU(name="stage1_unit3_relu3")(stage1_unit3_bn3)
    stage1_unit3_conv3 = Conv2D(filters=256, kernel_size=(1, 1), name="stage1_unit3_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage1_unit3_relu3)
    plus2 = Add()([stage1_unit3_conv3, plus1_v2])

    stage2_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit1_bn1", trainable=False)(plus2)
    stage2_unit1_relu1 = ReLU(name="stage2_unit1_relu1")(stage2_unit1_bn1)
    stage2_unit1_conv1 = Conv2D(filters=128, kernel_size=(1, 1), name="stage2_unit1_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit1_relu1)
    stage2_unit1_sc = Conv2D(filters=512, kernel_size=(1, 1), name="stage2_unit1_sc", strides=[2, 2], padding="VALID", use_bias=False)(stage2_unit1_relu1)
    stage2_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit1_bn2", trainable=False)(stage2_unit1_conv1)
    stage2_unit1_relu2 = ReLU(name="stage2_unit1_relu2")(stage2_unit1_bn2)
    stage2_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit1_relu2)
    stage2_unit1_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="stage2_unit1_conv2", strides=[2, 2], padding="VALID", use_bias=False)(stage2_unit1_conv2_pad)
    stage2_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit1_bn3", trainable=False)(stage2_unit1_conv2)
    stage2_unit1_relu3 = ReLU(name="stage2_unit1_relu3")(stage2_unit1_bn3)
    stage2_unit1_conv3 = Conv2D(filters=512, kernel_size=(1, 1), name="stage2_unit1_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit1_relu3)
    plus3 = Add()([stage2_unit1_conv3, stage2_unit1_sc])

    stage2_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit2_bn1", trainable=False)(plus3)
    stage2_unit2_relu1 = ReLU(name="stage2_unit2_relu1")(stage2_unit2_bn1)
    stage2_unit2_conv1 = Conv2D(filters=128, kernel_size=(1, 1), name="stage2_unit2_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit2_relu1)
    stage2_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit2_bn2", trainable=False)(stage2_unit2_conv1)
    stage2_unit2_relu2 = ReLU(name="stage2_unit2_relu2")(stage2_unit2_bn2)
    stage2_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit2_relu2)
    stage2_unit2_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="stage2_unit2_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit2_conv2_pad)
    stage2_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit2_bn3", trainable=False)(stage2_unit2_conv2)
    stage2_unit2_relu3 = ReLU(name="stage2_unit2_relu3")(stage2_unit2_bn3)
    stage2_unit2_conv3 = Conv2D(filters=512, kernel_size=(1, 1), name="stage2_unit2_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit2_relu3)
    plus4 = Add()([stage2_unit2_conv3, plus3])

    stage2_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit3_bn1", trainable=False)(plus4)
    stage2_unit3_relu1 = ReLU(name="stage2_unit3_relu1")(stage2_unit3_bn1)
    stage2_unit3_conv1 = Conv2D(filters=128, kernel_size=(1, 1), name="stage2_unit3_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit3_relu1)
    stage2_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit3_bn2", trainable=False)(stage2_unit3_conv1)
    stage2_unit3_relu2 = ReLU(name="stage2_unit3_relu2")(stage2_unit3_bn2)
    stage2_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit3_relu2)
    stage2_unit3_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="stage2_unit3_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit3_conv2_pad)
    stage2_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit3_bn3", trainable=False)(stage2_unit3_conv2)
    stage2_unit3_relu3 = ReLU(name="stage2_unit3_relu3")(stage2_unit3_bn3)
    stage2_unit3_conv3 = Conv2D(filters=512, kernel_size=(1, 1), name="stage2_unit3_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit3_relu3)
    plus5 = Add()([stage2_unit3_conv3, plus4])

    stage2_unit4_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit4_bn1", trainable=False)(plus5)
    stage2_unit4_relu1 = ReLU(name="stage2_unit4_relu1")(stage2_unit4_bn1)
    stage2_unit4_conv1 = Conv2D(filters=128, kernel_size=(1, 1), name="stage2_unit4_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit4_relu1)
    stage2_unit4_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit4_bn2", trainable=False)(stage2_unit4_conv1)
    stage2_unit4_relu2 = ReLU(name="stage2_unit4_relu2")(stage2_unit4_bn2)
    stage2_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit4_relu2)
    stage2_unit4_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="stage2_unit4_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit4_conv2_pad)
    stage2_unit4_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage2_unit4_bn3", trainable=False)(stage2_unit4_conv2)
    stage2_unit4_relu3 = ReLU(name="stage2_unit4_relu3")(stage2_unit4_bn3)
    stage2_unit4_conv3 = Conv2D(filters=512, kernel_size=(1, 1), name="stage2_unit4_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage2_unit4_relu3)
    plus6 = Add()([stage2_unit4_conv3, plus5])

    stage3_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit1_bn1", trainable=False)(plus6)
    stage3_unit1_relu1 = ReLU(name="stage3_unit1_relu1")(stage3_unit1_bn1)
    stage3_unit1_conv1 = Conv2D(filters=256, kernel_size=(1, 1), name="stage3_unit1_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit1_relu1)
    stage3_unit1_sc = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit1_sc", strides=[2, 2], padding="VALID", use_bias=False)(stage3_unit1_relu1)
    stage3_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit1_bn2", trainable=False)(stage3_unit1_conv1)
    stage3_unit1_relu2 = ReLU(name="stage3_unit1_relu2")(stage3_unit1_bn2)
    stage3_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit1_relu2)
    stage3_unit1_conv2 = Conv2D(filters=256, kernel_size=(3, 3), name="stage3_unit1_conv2", strides=[2, 2], padding="VALID", use_bias=False)(stage3_unit1_conv2_pad)
    ssh_m1_red_conv = Conv2D(filters=256, kernel_size=(1, 1), name="ssh_m1_red_conv", strides=[1, 1], padding="VALID", use_bias=True)(stage3_unit1_relu2)
    stage3_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit1_bn3", trainable=False)(stage3_unit1_conv2)
    ssh_m1_red_conv_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m1_red_conv_bn", trainable=False)(ssh_m1_red_conv)
    stage3_unit1_relu3 = ReLU(name="stage3_unit1_relu3")(stage3_unit1_bn3)
    ssh_m1_red_conv_relu = ReLU(name="ssh_m1_red_conv_relu")(ssh_m1_red_conv_bn)
    stage3_unit1_conv3 = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit1_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit1_relu3)
    plus7 = Add()([stage3_unit1_conv3, stage3_unit1_sc])

    stage3_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit2_bn1", trainable=False)(plus7)
    stage3_unit2_relu1 = ReLU(name="stage3_unit2_relu1")(stage3_unit2_bn1)
    stage3_unit2_conv1 = Conv2D(filters=256, kernel_size=(1, 1), name="stage3_unit2_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit2_relu1)
    stage3_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit2_bn2", trainable=False)(stage3_unit2_conv1)
    stage3_unit2_relu2 = ReLU(name="stage3_unit2_relu2")(stage3_unit2_bn2)
    stage3_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit2_relu2)
    stage3_unit2_conv2 = Conv2D(filters=256, kernel_size=(3, 3), name="stage3_unit2_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit2_conv2_pad)
    stage3_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit2_bn3", trainable=False)(stage3_unit2_conv2)
    stage3_unit2_relu3 = ReLU(name="stage3_unit2_relu3")(stage3_unit2_bn3)
    stage3_unit2_conv3 = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit2_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit2_relu3)
    plus8 = Add()([stage3_unit2_conv3, plus7])

    stage3_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit3_bn1", trainable=False)(plus8)
    stage3_unit3_relu1 = ReLU(name="stage3_unit3_relu1")(stage3_unit3_bn1)
    stage3_unit3_conv1 = Conv2D(filters=256, kernel_size=(1, 1), name="stage3_unit3_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit3_relu1)
    stage3_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit3_bn2", trainable=False)(stage3_unit3_conv1)
    stage3_unit3_relu2 = ReLU(name="stage3_unit3_relu2")(stage3_unit3_bn2)
    stage3_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit3_relu2)
    stage3_unit3_conv2 = Conv2D(filters=256, kernel_size=(3, 3), name="stage3_unit3_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit3_conv2_pad)
    stage3_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit3_bn3", trainable=False)(stage3_unit3_conv2)
    stage3_unit3_relu3 = ReLU(name="stage3_unit3_relu3")(stage3_unit3_bn3)
    stage3_unit3_conv3 = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit3_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit3_relu3)
    plus9 = Add()([stage3_unit3_conv3, plus8])

    stage3_unit4_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit4_bn1", trainable=False)(plus9)
    stage3_unit4_relu1 = ReLU(name="stage3_unit4_relu1")(stage3_unit4_bn1)
    stage3_unit4_conv1 = Conv2D(filters=256, kernel_size=(1, 1), name="stage3_unit4_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit4_relu1)
    stage3_unit4_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit4_bn2", trainable=False)(stage3_unit4_conv1)
    stage3_unit4_relu2 = ReLU(name="stage3_unit4_relu2")(stage3_unit4_bn2)
    stage3_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit4_relu2)
    stage3_unit4_conv2 = Conv2D(filters=256, kernel_size=(3, 3), name="stage3_unit4_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit4_conv2_pad)
    stage3_unit4_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit4_bn3", trainable=False)(stage3_unit4_conv2)
    stage3_unit4_relu3 = ReLU(name="stage3_unit4_relu3")(stage3_unit4_bn3)
    stage3_unit4_conv3 = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit4_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit4_relu3)
    plus10 = Add()([stage3_unit4_conv3, plus9])

    stage3_unit5_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit5_bn1", trainable=False)(plus10)
    stage3_unit5_relu1 = ReLU(name="stage3_unit5_relu1")(stage3_unit5_bn1)
    stage3_unit5_conv1 = Conv2D(filters=256, kernel_size=(1, 1), name="stage3_unit5_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit5_relu1)
    stage3_unit5_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit5_bn2", trainable=False)(stage3_unit5_conv1)
    stage3_unit5_relu2 = ReLU(name="stage3_unit5_relu2")(stage3_unit5_bn2)
    stage3_unit5_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit5_relu2)
    stage3_unit5_conv2 = Conv2D(filters=256, kernel_size=(3, 3), name="stage3_unit5_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit5_conv2_pad)
    stage3_unit5_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit5_bn3", trainable=False)(stage3_unit5_conv2)
    stage3_unit5_relu3 = ReLU(name="stage3_unit5_relu3")(stage3_unit5_bn3)
    stage3_unit5_conv3 = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit5_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit5_relu3)
    plus11 = Add()([stage3_unit5_conv3, plus10])

    stage3_unit6_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit6_bn1", trainable=False)(plus11)
    stage3_unit6_relu1 = ReLU(name="stage3_unit6_relu1")(stage3_unit6_bn1)
    stage3_unit6_conv1 = Conv2D(filters=256, kernel_size=(1, 1), name="stage3_unit6_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit6_relu1)
    stage3_unit6_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit6_bn2", trainable=False)(stage3_unit6_conv1)
    stage3_unit6_relu2 = ReLU(name="stage3_unit6_relu2")(stage3_unit6_bn2)
    stage3_unit6_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit6_relu2)
    stage3_unit6_conv2 = Conv2D(filters=256, kernel_size=(3, 3), name="stage3_unit6_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit6_conv2_pad)
    stage3_unit6_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage3_unit6_bn3", trainable=False)(stage3_unit6_conv2)
    stage3_unit6_relu3 = ReLU(name="stage3_unit6_relu3")(stage3_unit6_bn3)
    stage3_unit6_conv3 = Conv2D(filters=1024, kernel_size=(1, 1), name="stage3_unit6_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage3_unit6_relu3)
    plus12 = Add()([stage3_unit6_conv3, plus11])

    stage4_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit1_bn1", trainable=False)(plus12)
    stage4_unit1_relu1 = ReLU(name="stage4_unit1_relu1")(stage4_unit1_bn1)
    stage4_unit1_conv1 = Conv2D(filters=512, kernel_size=(1, 1), name="stage4_unit1_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit1_relu1)
    stage4_unit1_sc = Conv2D(filters=2048, kernel_size=(1, 1), name="stage4_unit1_sc", strides=[2, 2], padding="VALID", use_bias=False)(stage4_unit1_relu1)
    stage4_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit1_bn2", trainable=False)(stage4_unit1_conv1)
    stage4_unit1_relu2 = ReLU(name="stage4_unit1_relu2")(stage4_unit1_bn2)
    stage4_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit1_relu2)
    stage4_unit1_conv2 = Conv2D(filters=512, kernel_size=(3, 3), name="stage4_unit1_conv2", strides=[2, 2], padding="VALID", use_bias=False)(stage4_unit1_conv2_pad)
    ssh_c2_lateral = Conv2D(filters=256, kernel_size=(1, 1), name="ssh_c2_lateral", strides=[1, 1], padding="VALID", use_bias=True)(stage4_unit1_relu2)
    stage4_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit1_bn3", trainable=False)(stage4_unit1_conv2)
    ssh_c2_lateral_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_c2_lateral_bn", trainable=False)(ssh_c2_lateral)
    stage4_unit1_relu3 = ReLU(name="stage4_unit1_relu3")(stage4_unit1_bn3)
    ssh_c2_lateral_relu = ReLU(name="ssh_c2_lateral_relu")(ssh_c2_lateral_bn)
    stage4_unit1_conv3 = Conv2D(filters=2048, kernel_size=(1, 1), name="stage4_unit1_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit1_relu3)
    plus13 = Add()([stage4_unit1_conv3, stage4_unit1_sc])

    stage4_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit2_bn1", trainable=False)(plus13)
    stage4_unit2_relu1 = ReLU(name="stage4_unit2_relu1")(stage4_unit2_bn1)
    stage4_unit2_conv1 = Conv2D(filters=512, kernel_size=(1, 1), name="stage4_unit2_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit2_relu1)
    stage4_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit2_bn2", trainable=False)(stage4_unit2_conv1)
    stage4_unit2_relu2 = ReLU(name="stage4_unit2_relu2")(stage4_unit2_bn2)
    stage4_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit2_relu2)
    stage4_unit2_conv2 = Conv2D(filters=512, kernel_size=(3, 3), name="stage4_unit2_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit2_conv2_pad)
    stage4_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit2_bn3", trainable=False)(stage4_unit2_conv2)
    stage4_unit2_relu3 = ReLU(name="stage4_unit2_relu3")(stage4_unit2_bn3)
    stage4_unit2_conv3 = Conv2D(filters=2048, kernel_size=(1, 1), name="stage4_unit2_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit2_relu3)
    plus14 = Add()([stage4_unit2_conv3, plus13])

    stage4_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit3_bn1", trainable=False)(plus14)
    stage4_unit3_relu1 = ReLU(name="stage4_unit3_relu1")(stage4_unit3_bn1)
    stage4_unit3_conv1 = Conv2D(filters=512, kernel_size=(1, 1), name="stage4_unit3_conv1", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit3_relu1)
    stage4_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit3_bn2", trainable=False)(stage4_unit3_conv1)
    stage4_unit3_relu2 = ReLU(name="stage4_unit3_relu2")(stage4_unit3_bn2)
    stage4_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit3_relu2)
    stage4_unit3_conv2 = Conv2D(filters=512, kernel_size=(3, 3), name="stage4_unit3_conv2", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit3_conv2_pad)
    stage4_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name="stage4_unit3_bn3", trainable=False)(stage4_unit3_conv2)
    stage4_unit3_relu3 = ReLU(name="stage4_unit3_relu3")(stage4_unit3_bn3)
    stage4_unit3_conv3 = Conv2D(filters=2048, kernel_size=(1, 1), name="stage4_unit3_conv3", strides=[1, 1], padding="VALID", use_bias=False)(stage4_unit3_relu3)
    plus15 = Add()([stage4_unit3_conv3, plus14])

    bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn1", trainable=False)(plus15)
    relu1 = ReLU(name="relu1")(bn1)

    ssh_c3_lateral = Conv2D(filters=256, kernel_size=(1, 1), name="ssh_c3_lateral", strides=[1, 1], padding="VALID", use_bias=True)(relu1)
    ssh_c3_lateral_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_c3_lateral_bn", trainable=False)(ssh_c3_lateral)
    ssh_c3_lateral_relu = ReLU(name="ssh_c3_lateral_relu")(ssh_c3_lateral_bn)

    ssh_m3_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)
    ssh_m3_det_conv1 = Conv2D(filters=256, kernel_size=(3, 3), name="ssh_m3_det_conv1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_conv1_pad)
    ssh_m3_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)
    ssh_m3_det_context_conv1 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m3_det_context_conv1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_context_conv1_pad)
    ssh_c3_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_c3_up")(ssh_c3_lateral_relu)
    ssh_m3_det_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m3_det_conv1_bn", trainable=False)(ssh_m3_det_conv1)
    ssh_m3_det_context_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv1_bn", trainable=False)(ssh_m3_det_context_conv1)

    x1_shape = tf.shape(ssh_c3_up)
    x2_shape = tf.shape(ssh_c2_lateral_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop0 = tf.slice(ssh_c3_up, offsets, size, "crop0")

    ssh_m3_det_context_conv1_relu = ReLU(name="ssh_m3_det_context_conv1_relu")(ssh_m3_det_context_conv1_bn)
    plus0_v2 = Add()([ssh_c2_lateral_relu, crop0])

    ssh_m3_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m3_det_context_conv1_relu)
    ssh_m3_det_context_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m3_det_context_conv2", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_context_conv2_pad)
    ssh_m3_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m3_det_context_conv1_relu)
    ssh_m3_det_context_conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m3_det_context_conv3_1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_context_conv3_1_pad)
    ssh_c2_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus0_v2)
    ssh_c2_aggr = Conv2D(filters=256, kernel_size=(3, 3), name="ssh_c2_aggr", strides=[1, 1], padding="VALID", use_bias=True)(ssh_c2_aggr_pad)
    ssh_m3_det_context_conv2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv2_bn", trainable=False)(ssh_m3_det_context_conv2)
    ssh_m3_det_context_conv3_1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_1_bn", trainable=False)(ssh_m3_det_context_conv3_1)
    ssh_c2_aggr_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_c2_aggr_bn", trainable=False)(ssh_c2_aggr)
    ssh_m3_det_context_conv3_1_relu = ReLU(name="ssh_m3_det_context_conv3_1_relu")(ssh_m3_det_context_conv3_1_bn)
    ssh_c2_aggr_relu = ReLU(name="ssh_c2_aggr_relu")(ssh_c2_aggr_bn)

    ssh_m3_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m3_det_context_conv3_1_relu)
    ssh_m3_det_context_conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m3_det_context_conv3_2", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_context_conv3_2_pad)
    ssh_m2_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)
    ssh_m2_det_conv1 = Conv2D(filters=256, kernel_size=(3, 3), name="ssh_m2_det_conv1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_conv1_pad)
    ssh_m2_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)
    ssh_m2_det_context_conv1 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m2_det_context_conv1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_context_conv1_pad)
    ssh_m2_red_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_m2_red_up")(ssh_c2_aggr_relu)
    ssh_m3_det_context_conv3_2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_2_bn", trainable=False)(ssh_m3_det_context_conv3_2)
    ssh_m2_det_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m2_det_conv1_bn", trainable=False)(ssh_m2_det_conv1)
    ssh_m2_det_context_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv1_bn", trainable=False)(ssh_m2_det_context_conv1)

    x1_shape = tf.shape(ssh_m2_red_up)
    x2_shape = tf.shape(ssh_m1_red_conv_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop1 = tf.slice(ssh_m2_red_up, offsets, size, "crop1")

    ssh_m3_det_concat = concatenate([ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn], 3, name="ssh_m3_det_concat")
    ssh_m2_det_context_conv1_relu = ReLU(name="ssh_m2_det_context_conv1_relu")(ssh_m2_det_context_conv1_bn)
    plus1_v1 = Add()([ssh_m1_red_conv_relu, crop1])
    ssh_m3_det_concat_relu = ReLU(name="ssh_m3_det_concat_relu")(ssh_m3_det_concat)

    ssh_m2_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m2_det_context_conv1_relu)
    ssh_m2_det_context_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m2_det_context_conv2", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_context_conv2_pad)
    ssh_m2_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m2_det_context_conv1_relu)
    ssh_m2_det_context_conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m2_det_context_conv3_1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_context_conv3_1_pad)
    ssh_c1_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus1_v1)
    ssh_c1_aggr = Conv2D(filters=256, kernel_size=(3, 3), name="ssh_c1_aggr", strides=[1, 1], padding="VALID", use_bias=True)(ssh_c1_aggr_pad)

    face_rpn_cls_score_stride32 = Conv2D(filters=4, kernel_size=(1, 1), name="face_rpn_cls_score_stride32", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_concat_relu)
    inter_1 = concatenate([face_rpn_cls_score_stride32[:, :, :, 0], face_rpn_cls_score_stride32[:, :, :, 1]], axis=1)
    inter_2 = concatenate([face_rpn_cls_score_stride32[:, :, :, 2], face_rpn_cls_score_stride32[:, :, :, 3]], axis=1)
    face_rpn_cls_score_reshape_stride32 = tf.transpose(tf.stack([inter_1, inter_2]), (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride32")

    face_rpn_bbox_pred_stride32 = Conv2D(filters=8, kernel_size=(1, 1), name="face_rpn_bbox_pred_stride32", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_concat_relu)
    face_rpn_landmark_pred_stride32 = Conv2D(filters=20, kernel_size=(1, 1), name="face_rpn_landmark_pred_stride32", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m3_det_concat_relu)

    ssh_m2_det_context_conv2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv2_bn", trainable=False)(ssh_m2_det_context_conv2)
    ssh_m2_det_context_conv3_1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_1_bn", trainable=False)(ssh_m2_det_context_conv3_1)
    ssh_c1_aggr_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_c1_aggr_bn", trainable=False)(ssh_c1_aggr)
    ssh_m2_det_context_conv3_1_relu = ReLU(name="ssh_m2_det_context_conv3_1_relu")(ssh_m2_det_context_conv3_1_bn)
    ssh_c1_aggr_relu = ReLU(name="ssh_c1_aggr_relu")(ssh_c1_aggr_bn)

    face_rpn_cls_prob_stride32 = Softmax(name="face_rpn_cls_prob_stride32")(face_rpn_cls_score_reshape_stride32)
    input_shape = [tf.shape(face_rpn_cls_prob_stride32)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(
        tf.stack([face_rpn_cls_prob_stride32[:, 0:sz, :, 0], face_rpn_cls_prob_stride32[:, sz:, :, 0],
                  face_rpn_cls_prob_stride32[:, 0:sz, :, 1], face_rpn_cls_prob_stride32[:, sz:, :, 1]]),
        (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride32"
    )

    ssh_m2_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m2_det_context_conv3_1_relu)
    ssh_m2_det_context_conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m2_det_context_conv3_2", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_context_conv3_2_pad)
    ssh_m1_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)
    ssh_m1_det_conv1 = Conv2D(filters=256, kernel_size=(3, 3), name="ssh_m1_det_conv1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_conv1_pad)
    ssh_m1_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)
    ssh_m1_det_context_conv1 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m1_det_context_conv1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_context_conv1_pad)
    ssh_m2_det_context_conv3_2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_2_bn", trainable=False)(ssh_m2_det_context_conv3_2)
    ssh_m1_det_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m1_det_conv1_bn", trainable=False)(ssh_m1_det_conv1)
    ssh_m1_det_context_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv1_bn", trainable=False)(ssh_m1_det_context_conv1)

    ssh_m2_det_concat = concatenate([ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn], 3, name="ssh_m2_det_concat")
    ssh_m1_det_context_conv1_relu = ReLU(name="ssh_m1_det_context_conv1_relu")(ssh_m1_det_context_conv1_bn)
    ssh_m2_det_concat_relu = ReLU(name="ssh_m2_det_concat_relu")(ssh_m2_det_concat)

    ssh_m1_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m1_det_context_conv1_relu)
    ssh_m1_det_context_conv2 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m1_det_context_conv2", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_context_conv2_pad)
    ssh_m1_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m1_det_context_conv1_relu)
    ssh_m1_det_context_conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m1_det_context_conv3_1", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_context_conv3_1_pad)

    face_rpn_cls_score_stride16 = Conv2D(filters=4, kernel_size=(1, 1), name="face_rpn_cls_score_stride16", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_concat_relu)
    inter_1 = concatenate([face_rpn_cls_score_stride16[:, :, :, 0], face_rpn_cls_score_stride16[:, :, :, 1]], axis=1)
    inter_2 = concatenate([face_rpn_cls_score_stride16[:, :, :, 2], face_rpn_cls_score_stride16[:, :, :, 3]], axis=1)
    face_rpn_cls_score_reshape_stride16 = tf.transpose(tf.stack([inter_1, inter_2]), (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride16")

    face_rpn_bbox_pred_stride16 = Conv2D(filters=8, kernel_size=(1, 1), name="face_rpn_bbox_pred_stride16", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_concat_relu)
    face_rpn_landmark_pred_stride16 = Conv2D(filters=20, kernel_size=(1, 1), name="face_rpn_landmark_pred_stride16", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m2_det_concat_relu)

    ssh_m1_det_context_conv2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv2_bn", trainable=False)(ssh_m1_det_context_conv2)
    ssh_m1_det_context_conv3_1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_1_bn", trainable=False)(ssh_m1_det_context_conv3_1)
    ssh_m1_det_context_conv3_1_relu = ReLU(name="ssh_m1_det_context_conv3_1_relu")(ssh_m1_det_context_conv3_1_bn)

    face_rpn_cls_prob_stride16 = Softmax(name="face_rpn_cls_prob_stride16")(face_rpn_cls_score_reshape_stride16)
    input_shape = [tf.shape(face_rpn_cls_prob_stride16)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(
        tf.stack([face_rpn_cls_prob_stride16[:, 0:sz, :, 0], face_rpn_cls_prob_stride16[:, sz:, :, 0],
                  face_rpn_cls_prob_stride16[:, 0:sz, :, 1], face_rpn_cls_prob_stride16[:, sz:, :, 1]]),
        (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride16"
    )

    ssh_m1_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m1_det_context_conv3_1_relu)
    ssh_m1_det_context_conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), name="ssh_m1_det_context_conv3_2", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_context_conv3_2_pad)
    ssh_m1_det_context_conv3_2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_2_bn", trainable=False)(ssh_m1_det_context_conv3_2)

    ssh_m1_det_concat = concatenate([ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn], 3, name="ssh_m1_det_concat")
    ssh_m1_det_concat_relu = ReLU(name="ssh_m1_det_concat_relu")(ssh_m1_det_concat)

    face_rpn_cls_score_stride8 = Conv2D(filters=4, kernel_size=(1, 1), name="face_rpn_cls_score_stride8", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_concat_relu)
    inter_1 = concatenate([face_rpn_cls_score_stride8[:, :, :, 0], face_rpn_cls_score_stride8[:, :, :, 1]], axis=1)
    inter_2 = concatenate([face_rpn_cls_score_stride8[:, :, :, 2], face_rpn_cls_score_stride8[:, :, :, 3]], axis=1)
    face_rpn_cls_score_reshape_stride8 = tf.transpose(tf.stack([inter_1, inter_2]), (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride8")

    face_rpn_bbox_pred_stride8 = Conv2D(filters=8, kernel_size=(1, 1), name="face_rpn_bbox_pred_stride8", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_concat_relu)
    face_rpn_landmark_pred_stride8 = Conv2D(filters=20, kernel_size=(1, 1), name="face_rpn_landmark_pred_stride8", strides=[1, 1], padding="VALID", use_bias=True)(ssh_m1_det_concat_relu)

    face_rpn_cls_prob_stride8 = Softmax(name="face_rpn_cls_prob_stride8")(face_rpn_cls_score_reshape_stride8)
    input_shape = [tf.shape(face_rpn_cls_prob_stride8)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(
        tf.stack([face_rpn_cls_prob_stride8[:, 0:sz, :, 0], face_rpn_cls_prob_stride8[:, sz:, :, 0],
                  face_rpn_cls_prob_stride8[:, 0:sz, :, 1], face_rpn_cls_prob_stride8[:, sz:, :, 1]]),
        (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride8"
    )

    model = Model(
        inputs=data,
        outputs=[
            face_rpn_cls_prob_reshape_stride32, face_rpn_bbox_pred_stride32, face_rpn_landmark_pred_stride32,
            face_rpn_cls_prob_reshape_stride16, face_rpn_bbox_pred_stride16, face_rpn_landmark_pred_stride16,
            face_rpn_cls_prob_reshape_stride8,  face_rpn_bbox_pred_stride8,  face_rpn_landmark_pred_stride8,
        ],
    )

    model = _load_weights(model, weights_path)
    print("RetinaFace Loaded")
    return model


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_faces(
    img_path: Union[str, np.ndarray],
    threshold: float = 0.9,
    model: Optional[Model] = None,
    allow_upscaling: bool = True,
) -> Dict[str, Any]:
    resp = {}
    img = get_image(img_path)

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]
    _anchors_fpn = {
        "stride32": np.array([[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32),
        "stride16": np.array([[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32),
        "stride8":  np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }
    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    proposals_list = []
    scores_list = []
    landmarks_list = []

    im_tensor, im_info, im_scale = preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _, s in enumerate(_feat_stride_fpn):
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{s}"]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{s}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{s}"]
        anchors = anchors_plane(height, width, s, anchors_fpn).reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] *= bbox_stds[0]
        bbox_deltas[:, 1::4] *= bbox_stds[1]
        bbox_deltas[:, 2::4] *= bbox_stds[2]
        bbox_deltas[:, 3::4] *= bbox_stds[3]
        proposals = bbox_pred(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]
        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = landmark_pred(anchors, landmark_deltas)[order, :]
        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    if proposals.shape[0] == 0:
        return resp

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
    keep = cpu_nms(pre_det, nms_threshold)
    det = np.hstack((pre_det, proposals[:, 4:]))[keep, :]
    landmarks = landmarks[keep]

    for idx, face in enumerate(det):
        label = f"face_{idx + 1}"
        resp[label] = {
            "score": face[4],
            "facial_area": list(face[0:4].astype(int)),
            "landmarks": {
                "right_eye":   list(landmarks[idx][0]),
                "left_eye":    list(landmarks[idx][1]),
                "nose":        list(landmarks[idx][2]),
                "mouth_right": list(landmarks[idx][3]),
                "mouth_left":  list(landmarks[idx][4]),
            },
        }

    return resp


def extract_faces(
    img_path: Union[str, np.ndarray],
    threshold: float = 0.9,
    model: Optional[Model] = None,
    align: bool = True,
    allow_upscaling: bool = True,
    expand_face_area: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
    min_max_norm: bool = True,
) -> List[np.ndarray]:
    resp = []
    img = get_image(img_path)

    obj = detect_faces(img_path=img, threshold=threshold, model=model, allow_upscaling=allow_upscaling)
    if not isinstance(obj, dict):
        return resp

    for _, identity in obj.items():
        facial_area = identity["facial_area"]
        rotate_angle = 0
        rotate_direction = 1

        x = facial_area[0]
        y = facial_area[1]
        w = facial_area[2] - x
        h = facial_area[3] - y

        if expand_face_area > 0:
            expanded_w = w + int(w * expand_face_area / 100)
            expanded_h = h + int(h * expand_face_area / 100)
            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        facial_img = img[y: y + h, x: x + w]

        if align:
            aligned_img, rotate_angle, rotate_direction = alignment_procedure(
                img=img,
                left_eye=identity["landmarks"]["right_eye"],
                right_eye=identity["landmarks"]["left_eye"],
                nose=identity["landmarks"]["nose"],
            )
            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
                (x, y, x + w, y + h), rotate_angle, rotate_direction, (img.shape[0], img.shape[1])
            )
            facial_img = aligned_img[int(rotated_y1): int(rotated_y2), int(rotated_x1): int(rotated_x2)]

        if target_size is not None:
            facial_img = _pad_to_target(img=facial_img, target_size=target_size, min_max_norm=min_max_norm)

        facial_img = facial_img[:, :, ::-1]  # BGR to RGB
        resp.append(facial_img)

    return resp


def get_face(
    img_pillow: Any,
    model: Model,
    align: bool = True,
    allow_upscaling: bool = True,
    expand_face_area: int = 20,
) -> Optional[Any]:
    from PIL import Image as _Image
    img_array = np.array(img_pillow)[:, :, ::-1]  # PIL RGB → BGR numpy
    faces = extract_faces(
        img_path=img_array,
        model=model,
        align=align,
        allow_upscaling=allow_upscaling,
        expand_face_area=expand_face_area,
    )
    faces = [f for f in faces if f is not None and f.shape[0] > 0 and f.shape[1] > 0]
    if not faces:
        return None
    largest = max(faces, key=lambda x: x.shape[0] * x.shape[1])
    return _Image.fromarray(largest.astype('uint8'))
