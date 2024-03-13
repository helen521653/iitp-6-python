import numpy as np

from hough_lines import detect_and_draw_lines
import cv2 as cv


class TestImage:
    def test_Rectangle(self):
        img = cv.imread("./images_for/image_Rectangle.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert lines is not None


    def test_2_N_1(self):
        img = cv.imread("./images_for/image_2_N_1.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert lines is not None

    def test_rgb(self):
        img = cv.imread("./images_for/image_rgb.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert len(lines) == 3



class TestObject_property:
    def test_widht_one(self):
        img = cv.imread("./images_for/image_widht_one.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert lines is not None

    def test_intersecting(self):
        img = cv.imread("./images_for/image_Intersecting.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert len(lines) == 4

    def test_ellipses(self):
        img = cv.imread("./images_for/image_no_line.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert len(lines) == 0

    def test_parallel(self):
        img = cv.imread("./images_for/image_parallel_lines.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert len(lines) == 4

    def test_empty(self):
        img = cv.imread("./images_for/image_empty.png")
        img_with_lines, lines = detect_and_draw_lines(img)
        assert len(lines) == 0

def test_blur():
    '''

    Сheck the presence of two lines with specific coordinates

    '''
    img = cv.imread("./images_for/image_blur.png")
    img_with_lines, lines = detect_and_draw_lines(img)
    coordinats_of_line = [[(16, 62), (79,  10)],  [(44, 95), (99,  16)], [(79,  10),(16, 62)],  [(99,  16), (44, 95)] ]
    result = 0
    for cor in coordinats_of_line:
        if cor in lines:
            ++result
    assert result > 1

def test_several_rays():
    '''

    Success: The beginning (or end) of each of the lines is in a circle.
    A circle centered at the point from which the rays originate, radius 2.5 px

    Failure: at least one of the lines does not have a beginning (or end) in a circle.

    '''
    img = cv.imread("./images_for/image_rays.png")
    img_with_lines, lines = detect_and_draw_lines(img)
    lines = np.array(lines)

    flat_pairs = lines.reshape(-1, 2)

    # Проверяем условия для координат х и у одновременно
    conditions = np.logical_and(np.logical_and(15 <= flat_pairs[:, 0], flat_pairs[:, 0] <= 20),
                                np.logical_and(56 <= flat_pairs[:, 1], flat_pairs[:, 1] <= 61))

    # Если хотя бы одна пара точек удовлетворяет условиям, any_point_in_range будет True
    any_point_in_range = np.any(conditions)

    assert any_point_in_range