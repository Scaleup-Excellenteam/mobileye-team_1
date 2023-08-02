from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy import signal as sg
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = './test'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def convolve(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    return sg.convolve(img, kernel, "same")


def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> Tuple:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and your imagination to implement.

    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :param kwargs: Whatever config you want to pass in here.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """
    # Add blur to the image to reduce noise
    c_image = cv2.GaussianBlur(c_image, (3, 3), 0.65)

    # Convert the input RGB image to HSV color space
    hsv_image = cv2.cvtColor(c_image, cv2.COLOR_RGB2HSV)

    # Get the V channel from the HSV image
    v_channel = hsv_image[:, :, 2]

    # Apply convolution to enhance edges and intensity changes using the Laplacian kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    convolved_v_channel = convolve2d(v_channel, kernel, mode='same')

    # Threshold the convolved V channel to create a binary mask
    threshold = 100
    mask = (convolved_v_channel > threshold).astype(np.uint8)

    # Define the range of red and green colors in HSV space
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])

    # Create masks for red and green regions
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red) & mask
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green) & mask

    # Find the coordinates of red and green regions
    red_coords = np.argwhere(mask_red)
    green_coords = np.argwhere(mask_green)

    x_red, y_red = red_coords[:, 1], red_coords[:, 0]
    x_green, y_green = green_coords[:, 1], green_coords[:, 0]

    return x_red, y_red, x_green, y_green


# GIVEN CODE TO TEST YOUR IMPLEMENTATION AND PLOT THE PICTURES
def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'r'
            # plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str]=None, fig_num=None):
    """
    Run the attention code.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exist in your project then:
    print(DEFAULT_BASE_DIR)
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))

        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
