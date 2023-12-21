import numpy as np
import cv2
from matchPics import matchPics
import scipy
import matplotlib.pyplot as plt
from helper import plotMatches

#Read the image and convert to grayscale, if necessary
scipy.ndimage.rotate
cv_cover = cv2.imread('../data/cv_cover.jpg')

hist=[]
degree=[]
for i in range(1,36):
	#Rotate Image
    deg=i*10
    degree.append(deg)
    img_rot=scipy.ndimage.rotate(cv_cover,deg)
    #Compute features, descriptors and Match features
    matches,locs1,locs2=matchPics(cv_cover, img_rot)
    #plotMatches(cv_cover, img_rot, matches, locs1, locs2)

	#Update histogram
    hist.append(matches.shape[0])
    print("1")
print(hist)

#Display histogram
angles = np.arange(0, 35)
plt.bar(angles, hist)
plt.xlabel('Angle')
plt.ylabel('Match Count')
plt.title('Histogram of Match Counts for Each Angle')
plt.show()
