from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
import utils as ut
import pandas as pd
import consts as C
import crops as crp

DEFAULT_BASE_DIR: str = '../test3'
TFL_LABEL = ['traffic light']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]

result_df = pd.DataFrame(columns=[C.X, C.Y, C.COLOR, C.SEQ_IMAG, C.IMAG_PATH, C.GTIM_PATH])


def red_filter(hsv_image) -> np.ndarray:
    """
    Red filter for the image.
    :param hsv_image: The image itself as np.uint8, shape of (H, W, 3).
    :return: The filtered image.
    """
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([30, 255, 255])
    return cv2.inRange(hsv_image, lower_red, upper_red)


def green_filter(hsv_image) -> np.ndarray:
    """
    Green filter for the image.
    :param hsv_image: The image itself as np.uint8, shape of (H, W, 3).
    :return: The filtered image.
    """
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])
    return cv2.inRange(hsv_image, lower_green, upper_green)


def find_centroids(coordinates: np.ndarray, distance_threshold: int = 50) -> np.ndarray:
    """
    Find centroids of close points.
    :param coordinates: Array of (x, y) coordinates.
    :param distance_threshold: Maximum distance between points to consider them as part of the same cluster.
    :return: Array of (x, y) coordinates representing the centroids of close points.
    """
    if len(coordinates) == 0:
        return np.empty((0, 2), dtype=int)

    # Calculate pairwise distances between all points
    pairwise_distances = np.sqrt(np.sum((coordinates[:, None] - coordinates) ** 2, axis=-1))

    # Create a mask to identify points within the distance threshold
    close_points_mask = pairwise_distances < distance_threshold

    # Create an array to store the centroid coordinates
    centroids = []

    for mask in close_points_mask:
        if np.any(mask):
            # Find the mean coordinates of points within the distance threshold
            centroid = np.mean(coordinates[mask], axis=0)
            centroids.append(centroid)

    return np.array(centroids, dtype=int)


def add_to_df(lights, color, seq_img, image_path, image_json_path) -> None:
    """
    Add the lights to the result dataframe.
    :param red_lights: List of red lights coordinates.
    :param green_lights: List of green lights coordinates.
    :param seq_img: The sequence image number.
    :param image_path: The image path.
    :param image_json_path: The image json path.
    """

    global result_df

    for light in lights:
        df = {
            C.X: light[1],
            C.Y: light[0],
            C.COLOR: color,
            C.SEQ_IMAG: seq_img,
            C.IMAG_PATH: image_path,
            C.GTIM_PATH: image_json_path
        }
        result_df = result_df._append(df, ignore_index=True)


def find_tfl_lights(c_image: np.ndarray) -> Tuple:
    """
    Detect candidates for TFL lights.
    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """
    max_filtered_image = maximum_filter(c_image, 2)
    c_image = cv2.GaussianBlur(max_filtered_image, (3, 3), 0.2)
    hsv_image = cv2.cvtColor(c_image, cv2.COLOR_RGB2HSV)
    convolved_v_channel = convolve2d(hsv_image[:, :, 2], ut.kernel, mode='same')

    potential = (convolved_v_channel > 125).astype(np.uint8)
    red_coordinates = np.argwhere(potential & red_filter(hsv_image))
    green_coordinates = np.argwhere(potential & green_filter(hsv_image))

    red_lights = np.unique(find_centroids(red_coordinates), axis=0)
    green_lights = np.unique(find_centroids(green_coordinates), axis=0)

    return red_lights, green_lights


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


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None, seq_img=None):
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

    copy = np.ones(c_image.shape, dtype=np.uint8) * 255
    copy[int(c_image.shape[0] * ut.CROP_TOP):
                      int(c_image.shape[0] * ut.CROP_BOTTOM), :] = c_image[int(c_image.shape[0] * ut.CROP_TOP):
                      int(c_image.shape[0] * ut.CROP_BOTTOM), :]

    cropped = c_image[int(c_image.shape[0] * ut.CROP_TOP):
                      int(c_image.shape[0] * ut.CROP_BOTTOM), :]
    show_image_and_gt(copy, objects, fig_num)
    red_lights, green_lights = find_tfl_lights(copy)

    add_to_df(red_lights, C.RED, seq_img, image_path, image_json_path)
    add_to_df(green_lights, C.GREEN, seq_img, image_path, image_json_path)

    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_lights[:, 1], red_lights[:, 0], 'ro', markersize=4)
    plt.plot(green_lights[:, 1], green_lights[:, 0], 'go', markersize=4)


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
    tfls_df = pd.read_csv('../data/tfls.csv')

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

            matching_rows = tfls_df[tfls_df['imag_path'] == f"fullImages\\{image_path.split('/')[-1]}"]

            seq_img = matching_rows['seq_imag'].values[0]

            test_find_tfl_lights(image_path, image_json_path, seq_img=seq_img)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)

    print(result_df)
    crp.create_crops(result_df)



if __name__ == '__main__':
    main()
