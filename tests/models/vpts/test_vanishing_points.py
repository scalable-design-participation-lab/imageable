
from imageable._models.vpts.vpts_wrapper import VPTSWrapper
import numpy as np
from PIL import Image, ImageDraw
def test_vpts_wrapper_initialization():
    vpts_wrapper = VPTSWrapper()
    assert vpts_wrapper.model_name == "vpts"
    assert vpts_wrapper.is_loaded() is True


def _make_mock_building(
    img_size=(512, 512),
    front_pts=((150, 200), (300, 200), (300, 400), (150, 400)),
    side_pts=((300, 200), (380, 170), (380, 360), (300, 400)),
    bg_color=(200, 220, 255),
    front_color=(210, 210, 210),
    side_color=(180, 180, 180),
):
    """
    front_pts: 4 points (tl, tr, br, bl) of the front face in image coords
    side_pts: 4 points (tl, tr, br, bl) of the side face (in perspective)
    returns a PIL.Image
    """

    img = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(img)

    # fill faces
    draw.polygon(front_pts, fill=front_color, outline=(0,0,0))
    draw.polygon(side_pts, fill=side_color, outline=(0,0,0))

    # draw fake windows on front face (grid)
    rows, cols = 5, 3
    tl, tr, br, bl = front_pts
    for r in range(rows):
        for c in range(cols):
            # interpolate quad for each window cell
            # vertical interpolation
            y_top_left = tl[1] + (bl[1]-tl[1])*(r+0.2)/rows
            y_bot_left = tl[1] + (bl[1]-tl[1])*(r+0.8)/rows
            y_top_right = tr[1] + (br[1]-tr[1])*(r+0.2)/rows
            y_bot_right = tr[1] + (br[1]-tr[1])*(r+0.8)/rows
            # horizontal interpolation
            x_left_top = tl[0] + (tr[0]-tl[0])*(c+0.2)/cols
            x_right_top = tl[0] + (tr[0]-tl[0])*(c+0.8)/cols
            x_left_bot = bl[0] + (br[0]-bl[0])*(c+0.2)/cols
            x_right_bot = bl[0] + (br[0]-bl[0])*(c+0.8)/cols

            window_poly = [
                (x_left_top, y_top_left),
                (x_right_top, y_top_right),
                (x_right_bot, y_bot_right),
                (x_left_bot, y_bot_left),
            ]
            draw.polygon(window_poly, fill=(100,130,180))

    # draw windows on side face (skinnier, more foreshortened)
    rows_side, cols_side = 3, 1
    tl, tr, br, bl = side_pts
    for r in range(rows_side):
        for c in range(cols_side):
            y_top_left = tl[1] +(bl[1]-tl[1])*(r+0.2)/rows_side
            y_bot_left = tl[1] + (bl[1]-tl[1])*(r+0.8)/rows_side
            y_top_right = tr[1] + (br[1]-tr[1])*(r+0.2)/rows_side
            y_bot_right = tr[1] + (br[1]-tr[1])*(r+0.8)/rows_side

            x_left_top = tl[0] + (tr[0]-tl[0])*(c+0.2)/cols_side
            x_right_top = tl[0] + (tr[0]-tl[0])*(c+0.8)/cols_side
            x_left_bot = bl[0] + (br[0]-bl[0])*(c+0.2)/cols_side
            x_right_bot = bl[0] + (br[0]-bl[0])*(c+0.8)/cols_side

            window_poly = [
                (x_left_top, y_top_left),
                (x_right_top, y_top_right),
                (x_right_bot, y_bot_right),
                (x_left_bot, y_bot_left),
            ]
            draw.polygon(window_poly, fill=(90,110,150))

    return np.array(img)

def test_vpts_prediction():
    # Create a mock image (e.g., a blank image)
    img = _make_mock_building()
    vpts_wrapper = VPTSWrapper()

    vpts_2d = vpts_wrapper.predict(img)["vpts_2d"]
    assert vpts_2d.shape == (3, 2)


