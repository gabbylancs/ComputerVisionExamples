import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# - - - - - - - - - - - - - - - -  MAIN   FUNCTIONS  - - - - - - - - - - - - - - - - -

# function just to check the file stored is actually reading okay
def check_video_capture():
    # Create a VideoCapture object and read from input file
    cap = cv.VideoCapture('video.mp4')

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            resized = resize_frame(frame, 50)
            cv.imshow('Frame', resized)

            # Press Q on keyboard to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


# this will demonstrate the use of the lucas_kanade algorithm within the pipe
def lucas_kanade():
    print('here')
    cap = cv.VideoCapture('video.mp4')
    # params for corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=20,
                          blockSize=3)

    # Parameters for lucas-kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS |
                               cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, first_frame = cap.read()
    prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)
    prev_points = cv.goodFeaturesToTrack(prev_frame, mask=None,
                                         **feature_params,
                                         useHarrisDetector=1)  # move this into loop to look for features often
    print('here 2')
    while 1:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        current_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        current_points, st, err = cv.calcOpticalFlowPyrLK(prev_frame,
                                                          current_frame, prev_points, None, **lk_params)
        # Select good points
        if current_points is not None:
            good_current = current_points[st == 1]
            good_prev = prev_points[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_current, good_prev)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)),
                           color[i].tolist(), 2)  # Draw line at each point
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Draw circle on the keypoint
        img = cv.add(frame, mask)  # Add the lines/circles onto image
        resized = resize_frame(img, 50)
        cv.imshow('frame', resized)  # Display image
        cv.imwrite('OpticalFlow.png', resized)
        cv.waitKey(0)
        # break
        # Now update the previous frame and previous points
        prev_frame = current_frame.copy()
        prev_points = good_current.reshape(-1, 1, 2)
    cv.destroyAllWindows()


# OPTICAL FLOW MOD

# function - similar to lucas kanade method but modified to use the ORB fast feature detector as
# opposed to the (good features to track) function. The algorithm will also look for consistent
# features in 5 consecutive frames (modifiable) before they are defined as a movement. Also, once
# a feature has been missing for 3 frames (modifiable) it will be disregarded from the features to match. This
# function should produce an output of positions of features within known frames throughout the
# pipe movement - eventually with timestamps attached.

# consecutive_frames - the number of frames a feature must be present in to be counted
# missing_frames     - the number of frames a feature must be missing from to be discarded from features to look for
# return -> optical flow - (2 x n x m) array of features, the array of frames present in and the
# pixel position, (might be easier to make a library or something) - actually going to make a list
# of dictionaries

def optical_flow_mod(consecutive_frames=5, missing_frames=3):
    # create orb
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # create Des List & master matrix
    desList = []
    mm_shape = (2000, 100)
    masterMatrix_x = np.zeros(mm_shape)  # populate with x positions
    masterMatrix_y = np.zeros(mm_shape)  # populate with y positions

    # capture video from file:
    cap = cv.VideoCapture('video.mp4')

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find features in it
    ret, first_frame = cap.read()
    prev_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    kp_prev, des_prev = orb.detectAndCompute(prev_frame, None)

    # draw only key points location,not size and orientation
    # img2 = cv.drawKeypoints(prev_frame, kp_prev, None, color=(0, 255, 0), flags=0)
    # plt.imshow(img2), plt.show()

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)
    i = 0
    match_matrix = []
    while 1:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        current_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(current_frame, None)
        pts = cv.KeyPoint_convert(kp)

        # Match descriptors.
        matches = bf.match(des_prev, des)
        # Sort them in the order of their distance.
        matches_sorted = sorted(matches, key=lambda x: x.distance)

        for j in range(0, 20):
            match_des = des[matches_sorted[j].queryIdx]
            match_x = pts[matches_sorted[j].trainIdx][0]
            match_y = pts[matches_sorted[j].trainIdx][1]

            desList.append(match_des)
            masterMatrix_x[i * 20 + j][i] = match_x
            masterMatrix_y[i * 20 + j][i] = match_y

        i = i + 1
        if i >= 100:
            break

        # plt.imshow(img3), plt.show()


# - - - - - - - - - - - - - - - -  FUNCTIONS IN FUNCTIONS  - - - - - - - - - - - - - - - - -

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
