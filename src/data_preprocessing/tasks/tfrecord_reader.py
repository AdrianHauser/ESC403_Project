"""
Reader to parse TensorFlow Records Files, contains utility functions to augment data.
Provided by: https://www.kaggle.com/code/fantineh/data-reader-and-visualization
"""

import re
from pathlib import Path
from typing import Dict, List, Text, Tuple

import tensorflow as tf

from src.data_preprocessing.tasks.constants import (
    DATA_STATS,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
)


############# Utility Functions ###############
def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly axis-align crop input and output image tensors.

    Args:
      input_img: tensor with dimensions HWC.
      output_img: tensor with dimensions HWC.
      sample_size: side length (square) to crop to.
      num_in_channels: number of channels in input_img.
      num_out_channels: number of channels in output_img.
    Returns:
      input_img: tensor with dimensions HWC.
      output_img: tensor with dimensions HWC.
    """
    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(
        combined, [sample_size, sample_size, num_in_channels + num_out_channels]
    )
    input_img = combined[:, :, 0:num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img


def _get_base_key(key: Text) -> Text:
    """Extracts the base key from the provided key.

    Earth Engine exports TFRecords containing each data variable with its
    corresponding variable name. In the case of time sequences, the name of the
    data variable is of the form 'variable_1', 'variable_2', ..., 'variable_n',
    where 'variable' is the name of the variable, and n the number of elements
    in the time sequence. Extracting the base key ensures that each step of the
    time sequence goes through the same normalization steps.
    The base key obeys the following naming pattern: '([a-zA-Z]+)'
    For instance, for an input key 'variable_1', this function returns 'variable'.
    For an input key 'variable', this function simply returns 'variable'.

    Args:
      key: Input key.

    Returns:
      The corresponding base key.

    Raises:
      ValueError when `key` does not match the expected pattern.
    """
    match = re.match(r"([a-zA-Z]+)", key)
    if match:
        return match.group(1)
    raise ValueError(
        "The provided key does not match the expected pattern: {}".format(key)
    )


def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and rescales inputs with the stats corresponding to `key`.

    Args:
      inputs: Inputs to clip and rescale.
      key: Key describing the inputs.

    Returns:
      Clipped and rescaled input.

    Raises:
      ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(
            "No data statistics available for the requested key: {}.".format(key)
        )
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and normalizes inputs with the stats corresponding to `key`.

    Args:
      inputs: Inputs to clip and normalize.
      key: Key describing the inputs.

    Returns:
      Clipped and normalized input.

    Raises:
      ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(
            "No data statistics available for the requested key: {}.".format(key)
        )
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)


def _get_features_dict(
    sample_size: int,
    features: List[Text],
) -> Dict[Text, tf.io.FixedLenFeature]:
    """Creates a features dictionary for TensorFlow IO.

    Args:
      sample_size: Size of the input tiles (square).
      features: List of features names.

    Returns:
      A features dictionary for TensorFlow IO.
    """
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [
        tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32) for _ in features
    ]
    return dict(zip(features, columns))


def _parse_fn(
    example_proto: tf.train.Example,
    data_size: int,
    sample_size: int,
    num_in_channels: int,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads a serialized example.

    Args:
      example_proto: A TensorFlow example protobuf.
      data_size: Size of tiles (square) as read from input files.
      sample_size: Size the tiles (square) when input into the model.
      num_in_channels: Number of input channels.
      clip_and_normalize: True if the data should be clipped and normalized.
      clip_and_rescale: True if the data should be clipped and rescaled.
      random_crop: True if the data should be randomly cropped.

    Returns:
      (input_img, output_img) tuple of inputs and outputs to the ML model.
    """

    input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = _get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [
            _clip_and_normalize(features.get(key), key) for key in input_features
        ]
    elif clip_and_rescale:
        inputs_list = [
            _clip_and_rescale(features.get(key), key) for key in input_features
        ]
    else:
        inputs_list = [features.get(key) for key in input_features]

    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, "outputs_list should not be empty"
    outputs_stacked = tf.stack(outputs_list, axis=0)

    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, (
        "outputs_stacked should be rank 3"
        "but dimensions of outputs_stacked"
        f" are {outputs_stacked_shape}"
    )
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(
            input_img, output_img, sample_size, num_in_channels, 1
        )

    return input_img, output_img


def get_dataset(
    file_pattern: Text,
    data_size: int,
    sample_size: int,
    batch_size: int,
    num_in_channels: int,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
) -> tf.data.Dataset:
    """Gets the dataset from the file pattern.

    Args:
      file_pattern: Input file pattern.
      data_size: Size of tiles (square) as read from input files.
      sample_size: Size the tiles (square) when input into the model.
      batch_size: Batch size.
      num_in_channels: Number of input channels.
      clip_and_normalize: True if the data should be clipped and normalized, False
        otherwise.
      clip_and_rescale: True if the data should be clipped and rescaled, False
        otherwise.
      random_crop: True if the data should be randomly cropped.

    Returns:
      A TensorFlow dataset loaded from the input file pattern, with features
      described in the constants, and with the shapes determined from the input
      parameters to this function.
    """
    if clip_and_normalize and clip_and_rescale:
        raise ValueError("Cannot have both normalize and rescale.")

    if isinstance(file_pattern, Path):
        file_pattern = str(file_pattern)  # Convert Path object to string
        print(file_pattern)

    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_fn(  # pylint: disable=g-long-lambda
            x,
            data_size,
            sample_size,
            num_in_channels,
            clip_and_normalize,
            clip_and_rescale,
            random_crop,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
