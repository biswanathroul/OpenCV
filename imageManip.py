import math

import numpy as np
from PIL import Image
from skimage import color, io

import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    
    # Use skimage io.imread
    out = io.imread(image_path)
    pass
    

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out



def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    
    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols, :]
    pass
   

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    
    
    out = 0.5 * (image ** 2)
    pass
    

    return out



def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    
    output_image = np.zeros(shape=(output_rows, output_cols, 3))



  
    for x in range(output_rows):  # i for rows
        for y in range(output_cols):  # j for columns
            x_input = int(x * input_rows/output_rows)
            y_input = int(y * input_cols/output_cols)
            output_image[x][y] = input_image[x_input][y_input]
    pass

    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    
    x_coordinate, y_coordinate = point  
    output_y = (x_coordinate)*(np.sin(theta)) + (y_coordinate)*(np.cos(theta))
    output_x = (x_coordinate)*(np.cos(theta)) - (y_coordinate)*(np.sin(theta))
    
    return np.array([output_x, output_y])
    pass



def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    output_image = np.zeros_like(input_image)

    y = input_cols/2
    x = input_rows/2
    for row in range(input_rows):
        for column in range(input_cols):
            y_dash = int(((row - x)*np.sin(theta) + (column - y)*np.cos(theta)) + x)
            x_dash = int(((row - x) *np.cos(theta) - (column - y)*np.sin(theta)) + y)
            if 0<=x_dash<=input_rows and 0<=y_dash<=input_cols:
                output_image[row][column] = input_image[x_dash][y_dash]
    pass
   

    return output_image
