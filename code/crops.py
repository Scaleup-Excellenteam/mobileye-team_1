from pathlib import Path
from typing import Dict, Any, List
import cv2
import json
import consts as C
from pandas import DataFrame
from PIL import Image
from shapely.geometry import Polygon
import numpy as np
import utils as ut


SEQ: str = 'seq'  # The image seq number -> for tracing back the original image
IS_TRUE: str = 'is_true'  # Is it a traffic light or not.
IGNOR: str = 'is_ignore'  # If it's an unusual crop (like two tfl's or only half etc.) that you can just ignor it and
# investigate the reason after
CROP_PATH: str = 'path'
X0: str = 'x0'  # The bigger x value (the right corner)
X1: str = 'x1'  # The smaller x value (the left corner)
Y0: str = 'y0'  # The smaller y value (the lower corner)
Y1: str = 'y1'  # The bigger y value (the higher corner)
COL: str = 'col'
SEQ_IMAG: str = 'seq_imag'  # Serial number of the image
GTIM_PATH: str = 'gtim_path'
JSON_PATH: str = 'json_path'
IMAG_PATH: str = 'img_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COL]

# Files path
BASE_SNC_DIR: Path = Path.cwd().parent
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
CROP_DIR: Path = DATA_DIR / 'crops'
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'

CROP_CSV_NAME: str = 'crop_results.csv'  # result CSV name


def make_crop(*args, **kwargs):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """

    img_path = kwargs['img_path']
    x_coord = kwargs['x']
    y_coord = kwargs['y']
    # check the coords
    x0, x1 = x_coord + 20, x_coord - 20
    y0, y1 = (y_coord + 60, y_coord - 25) if kwargs['color'] == 'r' \
        else (y_coord + 25, y_coord - 60)

    # load the image
    img = Image.open(img_path)
    c_image = np.array(img)

    cropped_image = c_image[y1: y0, x1: x0]

    return x0, x1, y0, y1, cropped_image


def check_crop(*args, **kwargs):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """
    json_path = args[0]
    x0, x1, y0, y1 = args[1], args[2], args[3], args[4]

    rectangle_coords = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]  # Rectangle vertices
    polygons = []  # List to store Polygon objects

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract the objects list
    objects = data.get("objects", [])

    for item in objects:
        label = item.get("label", "")

        if label == "traffic light":
            polygon_coords = item.get("polygon", [])
            polygon = Polygon(polygon_coords)  # Create Polygon object
            polygons.append(polygon)

    # Check if the rectangle intersects with any of the polygons
    rectangle = Polygon(rectangle_coords)

    for polygon in polygons:
        if polygon.exterior.intersects(rectangle):
            return True, False

    return False, False


def save_for_part_2(crops_df: DataFrame):
    """
    *** No need to touch this. ***
    Saves the result DataFrame containing the crops data in the relevant folder under the relevant name for part 2.
    """
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()
    crops_sorted: DataFrame = crops_df.sort_values(by=SEQ)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!
    # Run this from your 'code' folder so that it will be in the right relative folder from your data folder.

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        x0, x1, y0, y1, crop = make_crop(img_path=row[IMAG_PATH], x=row[X], y=row[Y], color=row[COLOR])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = f'../data/crops/crop{index}{row[SEQ_IMAG]}.png'
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        result_template[CROP_PATH] = crop_path
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(row[GTIM_PATH], x0, x1, y0, y1)

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)

    # A Short function to help you save the whole thing - your welcome ;)
    save_for_part_2(result_df)
    return result_df
