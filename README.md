# convolution-with-tensorflow
# Simple TensorFlow Convolution Example

This repository contains a basic example of how to implement a 2D convolutional layer using TensorFlow's Keras API.

## Overview

The Python script `convolution_example.py` demonstrates the creation and application of a `tf.keras.layers.Conv2D` layer. It defines an input tensor simulating a batch of grayscale images and then applies a convolutional layer to it. The shapes of the input and output tensors are printed to the console.

## Code Description

The `convolution_example.py` script performs the following steps:

1.  **Imports TensorFlow:**
    ```python
    import tensorflow as tf
    ```

2.  **Defines Input Shape:**
    ```python
    input_shape = (32, 28, 28, 1)
    ```
    This line defines the shape of the input tensor. It assumes a batch of 32 grayscale images, each with a height of 28 pixels and a width of 28 pixels. The `1` represents the single color channel (grayscale).

3.  **Creates an Input Tensor:**
    ```python
    input_tensor = tf.random.normal(input_shape)
    ```
    A random normal tensor with the defined `input_shape` is created to simulate input data.

4.  **Defines Convolutional Layer Parameters:**
    ```python
    filters = 32
    kernel_size = (3, 3)
    strides = (1, 1)
    padding = 'same'
    activation = 'relu'
    ```
    These lines define the hyperparameters of the convolutional layer:
    * `filters`: The number of output filters (convolutional kernels), set to 32.
    * `kernel_size`: The height and width of the convolutional kernel, set to 3x3.
    * `strides`: The stride of the convolution along the height and width, set to 1 in both directions.
    * `padding`: The padding strategy, set to `'same'` to ensure the output has the same spatial dimensions as the input (when `strides` is 1).
    * `activation`: The activation function applied after the convolution, set to ReLU (Rectified Linear Unit).

5.  **Creates the Convolutional Layer:**
    ```python
    conv_layer = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation
    )
    ```
    A `Conv2D` layer is instantiated with the defined parameters.

6.  **Applies the Convolutional Layer:**
    ```python
    output_tensor = conv_layer(input_tensor)
    ```
    The `conv_layer` is applied to the `input_tensor`, resulting in the `output_tensor`.

7.  **Prints Tensor Shapes:**
    ```python
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)
    ```
    The shapes of the input and output tensors are printed to the console, demonstrating the effect of the convolutional operation on the tensor's dimensions.

## Getting Started

1.  **Prerequisites:**
    * Python 3.x
    * TensorFlow (install using `pip install tensorflow`)

2.  **Running the Script:**
    1.  Save the code as `convolution_example.py`.
    2.  Open a terminal or command prompt.
    3.  Navigate to the directory where you saved the file.
    4.  Run the script using the command: `python convolution_example.py`

## Output

Running the script will produce output similar to the following:
