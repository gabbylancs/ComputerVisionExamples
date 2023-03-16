import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt

focalLength = 800  # might be more like 0.07 - 0.4
diameter = 0.085


# - - - - - - - - main functions  - - - - - - - - - - - - - - - - - -

def optical_flow_mod(consecutive_frames=5, missing_frames=3):
    # create orb
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # capture video from file:
    cap = cv.VideoCapture('video.mp4')

    # Take first frame and find features in it
    ret, first_frame = cap.read()
    prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    kp_prev, des_prev = orb.detectAndCompute(prev_frame, None)
    pts_prev = cv.KeyPoint_convert(kp_prev)

    height, width, layers = first_frame.shape
    fourcc = cv .VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter('video.avi', fourcc, 1, (int(width/2), int(height/2) ))

    i = 0
    total_d = 0
    while 1:
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        # lazy way of getting it to do every 5th frame
        circle_x, circle_y = findCircleCenter(frame)  # circles_det = [x,y,r]
        # Create a mask image for drawing purposes
        mask = np.zeros_like(first_frame)
        if not ret:
            print('No frames grabbed!')
            break
        current_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(current_frame, None)
        pts = cv.KeyPoint_convert(kp)

        # Match descriptors.
        matches = bf.match(des_prev, des)  # (query, train)
        # Sort them in the order of their distance.
        matches_sorted = sorted(matches, key=lambda x: x.distance)
        avg_d = 0
        # we're going to only look at the top twenty matches
        for j in range(0, 10):
            match_des = des[matches_sorted[j].trainIdx]
            match_x = int(pts[matches_sorted[j].trainIdx][0])
            match_y = int(pts[matches_sorted[j].trainIdx][1])
            match_x_q = int(pts_prev[matches_sorted[j].queryIdx][0])
            match_y_q = int(pts_prev[matches_sorted[j].queryIdx][1])

            d = findPositions(match_x, match_y, match_x_q, match_y_q, int(circle_x), int(circle_y))
            if 3 > d > -3:
                avg_d += d

            cv.line(mask, (match_x_q, match_y_q), (match_x, match_y),
                    (0, 255, 0), 7)

        avg_d = avg_d/10
        total_d += avg_d

        # onto the next frame but draw current frame:

        img = cv.add(frame, mask)  # Add the lines/circles onto image
        resized = resize_frame(img, 50)
        cv.putText(resized, 'distance ' + str(total_d), (10, 450), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv.LINE_AA)
        video.write(resized)
        cv.imshow('frame', resized)  # Display image
        cv.waitKey(0)

        pts_prev = pts
        des_prev = des
        i = i + 1
        if i >= 100:
            video.release()
            break


# - - - - - - - - additional functions - - - - - - - - - - - - - - -

# RESIZE FRAME
# simple function to resize an image or frame
# img    - image/frame to be resized
# scale  - percentage to be scaled e.g. 50 to half the size, 200 to double
# return - resized, the resized image/frame
def resize_frame(img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized


# FIND CIRCLE CENTER
# function - to locate the pixel co-ordinates of the center of the pipe from the
# perspective of the camera. PRETTY USELESS FUNCTION ATM COULD BE WORKED ON TO LOOK AT
# THE SHAPE OF THE 'DARK CIRCLE' TO INFER ORIENTATION
# image -> image to be processed
def findCircleCenter(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    #  if the pixel intensity is < thresh value it will be set to zero
    dilated = cv.dilate(thresh, None, (7, 7), iterations=3)
    dilated = cv.bitwise_not(dilated)
    # resize_frame('img', dilated)
    # outline = cv2.morphologyEx(dilated, cv2.MORPH_GRADIENT, (65,65))
    eroded = cv.erode(dilated, None, (7, 7), iterations=3)
    canny = cv.Canny(eroded, 100, 200)
    # resize_frame('eroded', canny)
    rows = dilated.shape[0]
    circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, rows / 2, param1=30, param2=1, minRadius=5, maxRadius=200)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(image, center, radius, (255, 0, 255), 3)
    resized = resize_frame(image, 50)
    cv.imshow('frame', resized)  # Display image
    cv.waitKey(0)
    return center[0], center[1]


def findPositions(pt1s_x, pt1s_y, pt2s_x, pt2s_y, center_x, center_y):
    x_pix1 = pt1s_x - center_x
    y_pix1 = pt1s_y - center_y
    x_pix2 = pt2s_x - center_x
    y_pix2 = pt2s_y - center_y

    r_sq1 = np.square(x_pix1) + np.square(y_pix1)
    r_pix1 = np.sqrt(r_sq1)
    r_sq2 = np.square(x_pix2) + np.square(y_pix2)
    r_pix2 = np.sqrt(r_sq2)

    # okay so now we have the pixel distance of a 2.5cm object (radius of pipe)

    # now we're going to use the formula: d = (W*f)/w
    # where d = distance to object, W = width in cm, f = focal length and w = width in pixels

    d1 = (2.5 * focalLength) / r_pix1
    d2 = (2.5 * focalLength) / r_pix2

    delta_d = d2 - d1
    return delta_d
