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

    # Apply max filter to remove false positive detections
    max_filtered_image = ndimage.maximum_filter(c_image, 2)
    # cv2.imshow("max", max_filtered_image)

    # Add blur to the image to reduce noise
    c_image = cv2.GaussianBlur(max_filtered_image, (3, 3), 0.5)

    # Convert the input RGB image to HSV color space
    hsv_image = cv2.cvtColor(c_image, cv2.COLOR_RGB2HSV)
    # cv2.imshow("hsv", hsv_image)

    # Get the V channel from the HSV image
    v_channel = hsv_image[:, :, 2]
    # cv2.imshow("channel", v_channel)

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
    upper_red = np.array([30, 255, 255])
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

    c_image_copy = c_image.copy()
    draw_rectangles_around_points(c_image_copy, x_red, y_red, x_green, y_green)


    # Show the image with rectangles
    plt.imshow(c_image_copy)
    plt.plot(x_red, y_red, 'ro', markersize=4)
    plt.plot(x_green, y_green, 'go', markersize=4)

    return x_red, y_red, x_green, y_green


def non_max_suppression(rectangles, overlap_threshold=0):
    """
    Perform non-maximum suppression on the list of rectangles.

    :param rectangles: List of rectangles as (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
    :param overlap_threshold: Threshold to decide whether two rectangles should be merged.
    :return: List of rectangles after non-maximum suppression.
    """
    if len(rectangles) == 0:
        return []

    rectangles = np.array(rectangles)

    x1 = rectangles[:, 0]
    y1 = rectangles[:, 1]
    x2 = rectangles[:, 2]
    y2 = rectangles[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    sorted_indexes = np.argsort(y2)

    merged_rectangles = []

    while len(sorted_indexes) > 0:
        last = len(sorted_indexes) - 1
        i = sorted_indexes[last]
        merged_rectangles.append(rectangles[i])

        xx1 = np.maximum(x1[i], x1[sorted_indexes[:last]])
        yy1 = np.maximum(y1[i], y1[sorted_indexes[:last]])
        xx2 = np.minimum(x2[i], x2[sorted_indexes[:last]])
        yy2 = np.minimum(y2[i], y2[sorted_indexes[:last]])

        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)

        intersection_area = width * height
        union_area = area[i] + area[sorted_indexes[:last]] - intersection_area

        iou = intersection_area / union_area

        indexes_to_keep = np.where(iou <= overlap_threshold)[0]
        sorted_indexes = sorted_indexes[indexes_to_keep]

    return merged_rectangles


def draw_rectangles_around_points(image: np.ndarray, red_x_coords: List[int], red_y_coords: List[int],
                                 green_x_coords: List[int], green_y_coords: List[int]):
    """
    Draw rectangles around points with a specified width and height.

    :param image: The input image as a numpy ndarray.
    :param red_x_coords: The x-coordinates of the red points.
    :param red_y_coords: The y-coordinates of the red points.
    :param green_x_coords: The x-coordinates of the green points.
    :param green_y_coords: The y-coordinates of the green points.
    """
    red_rectangles = []
    green_rectangles = []

    for x, y in zip(red_x_coords, red_y_coords):
        top_left_x = x - 15
        top_left_y = y - 20
        bottom_right_x = x + 15
        bottom_right_y = y + 60

        red_rectangles.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    for x, y in zip(green_x_coords, green_y_coords):
        top_left_x = x - 15
        top_left_y = y - 60
        bottom_right_x = x + 15
        bottom_right_y = y + 20

        green_rectangles.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    # Apply non-maximum suppression to merge intersecting rectangles
    red_rectangles = non_max_suppression(red_rectangles)
    green_rectangles = non_max_suppression(green_rectangles)

    for rect in red_rectangles:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)

    for rect in green_rectangles:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)




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


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None):
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
