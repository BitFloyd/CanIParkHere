from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial.distance import cdist

import image_processor.config as config


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)
    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    intersections = [i[0] for i in intersections]

    return intersections


def get_hough_lines_from_image(im):
    '''
    Takes an image and returns all the detected houghlines in it.
    :param im: image_array (RGB)
    :return: detected lines (rho,theta)
    '''
    image_dim_low = min(im.shape[0], im.shape[1])
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    threshold_value, otsu_th_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(otsu_th_image, config.GHLFI_EDGES_MINVAL, config.GHLFI_EDGES_MAXVAL,
                      config.GHLFI_EDGES_APERTURSIZE)
    lines = cv2.HoughLines(edges, config.GHLFI_LINES_RHO, np.pi / config.GHLFI_LINES_THETA_DIVIDE,
                           int(config.GHLFI_LINES_THRESHOLD * image_dim_low))
    return lines

def save_intersection_lines_image(image,intersections,filename):

    for intersection in intersections:
        x = intersection[0]
        y = intersection[1]
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 255), thickness=-1)

    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_kmeans_coords_on_image(image,intersections,filename):

    for intersection in intersections:
        x = intersection[0]
        y = intersection[1]
        image = cv2.circle(image, (x, y), radius=15, color=(255, 0, 255), thickness=-1)

    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_perspective_transformed_on_image(image,filename):
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_hough_lines_image(image,lines,filename):

    for line in lines:
        rho,theta = line[0][0],line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite(filename,cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

def get_valid_perspective_from_localized_sign(image):
    '''

    :param image: the localized cropped sign (ndarray)
    :return: valid perspective transformed image of the sign.
    '''

    h, w, c = image.shape
    lines = get_hough_lines_from_image(image)

    if (lines is None):
        return None

    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)

    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (default_criteria_type, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    attempts = 10
    pts = np.array(intersections).astype(np.float32)
    # run kmeans on the coords
    _, centers = cv2.kmeans(pts, 4, None, criteria, attempts, flags)[1:]
    center_points = []
    for center in centers:
        x, y = center
        x = max(x, 0)
        y = max(y, 0)
        center_points.append([x, y])

    center_points = np.float32(center_points)
    frame_extrema_points = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    dist_matrix = cdist(center_points, frame_extrema_points)

    points_matched = np.array([frame_extrema_points[np.argmin(i)] for i in dist_matrix])
    M = cv2.getPerspectiveTransform(center_points, points_matched)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold_value, otsu_th_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dst = cv2.warpPerspective(otsu_th_image, M, (w, h))

    return dst
