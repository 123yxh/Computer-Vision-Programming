import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


def matchPics(I1, I2):
	ratio = 0.8  # 'ratio for BRIEF feature descriptor']
	sigma = 0.15  # 'threshold for corner detection using FAST feature detector'

	# Convert Images to GrayScale
	I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

	locs1 = corner_detection(I1, sigma)
	# Detect Features in Both Images
	locs2 = corner_detection(I2, sigma)
	# Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1, locs1)

	# Match features using the descritors
	desc2, locs2 = computeBrief(I2, locs2)

	# Match features using the descritors
	matches = briefMatch(desc1, desc2, ratio)

	return matches, locs1, locs2
