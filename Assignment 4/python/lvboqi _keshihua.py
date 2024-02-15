from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import create_filterbank

#择第一组滤波器
def extract_filter_responses(I, filterBank):
    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I[:, :, np.newaxis], (1, 1, 3))

    I_lab = rgb2lab(I)
    filterResponses = []
    for i in [0, 5, 10, 15]:  # Indices of the first filter in each group
        for j in range(3):  # For each channel
            I_filt = imfilter(I_lab[:,:,j], filterBank[i])
            filterResponses.append(I_filt)

    return filterResponses

# test
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from scipy.ndimage import convolve
from skimage.io import imread

# Load a sample image from skimage
image = imread('../data/desert/sun_avtmqreqkvhyoead.jpg')

# Define a filter bank
filter_bank = create_filterbank()

# Extract filter responses using the modified function
responses = extract_filter_responses(image, filter_bank)

# Filter types and channel names
filter_types = ['Gaussian', 'LoG', 'dx', 'dy']
channel_names = ['Red', 'Green', 'Blue']

# Create subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 20),
                         gridspec_kw={'left': 0.15, 'top': 0.85}) # Adjust layout

# Loop over the filter responses and plot them
for i in range(4):  # 4 filter groups
    for j in range(3):  # 3 channels (R, G, B)
        response_index = i * 3 + j
        axes[i, j].imshow(responses[response_index], cmap='gray')
        axes[i, j].axis('off')

        # Add titles to the top row
        if i == 0:
            axes[i, j].set_title(channel_names[j])

# Add labels for filter types on the left
for i in range(4):
    # Calculate the y position to align with the center of each row
    y_pos = 0.75 - i * 0.2
    fig.text(0.05, y_pos, filter_types[i], ha='center', va='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.show()