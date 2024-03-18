import numpy as np
import cv2

# Function to perform edge detection using Canny algorithm
def canny_edge_detection(image, low_threshold, high_threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return canny_edges

# Function to apply region of interest mask
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to detect lane lines using Hough transform
def hough_transform(image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_image

# Function to average slope and intercept of lines
def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1
