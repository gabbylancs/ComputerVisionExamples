#  @author: Gabby
#  Created: 23/03/23
#  Last updated: 23/02/2023

# The purpose of this document is to demonstrate the use of different algorithms on pipe images and or
# videos to compare their effectiveness and help decide which will be best used in the project.

# Run this file to run the entire project

import cv2
import numpy as np
import os

import Functions
import Definitions


# - - - - - - - - - - - - - - - - - MAIN SCRIPT - - - - - - - - - - - - - - - - - - - - - - - - - -
#   The main script will run through with the user the different options to be chosen and run them
#   accordingly.

Definitions.init()

interaction_unfinished = 1
while interaction_unfinished == 1:

    print("Would you like to (1) show the video, (2) use basic lucas-kanade algorithm, (3) try mod optical flow")
    selection_1 = input("please enter '1' or '2'")

    if selection_1 == '1':
        print('1 selected')
        Functions.check_video_capture()
    elif selection_1 == '2':
        print('2 Selected')
        Functions.lucas_kanade()
        break
    elif selection_1 == '3':
        Functions.optical_flow_mod()
        break
    else:
        print("That is not a valid response.")





