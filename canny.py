from PIL import Image
from scipy.signal import convolve
from scipy import ndimage
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def GaussFilter(n, sigma):
    '''
    Returns a Gaussian Mask of size 2(n+1)
    '''
    r = range(-n,n+1)
    return [(1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-(x**2)/(2*sigma**2)) for x in r]

def Fill(img, mag, high, low, i, j):
    '''
    Recursively go through the image to identify candidates for Hysteresis thresholding.
    Since this is only initially called when a pixel is > high,
    all pixels above low that are reached pass the threshold.
    '''
    if(img[i][j] == 1):
        return
    if mag[i][j] > low:
        img[i][j] = 1
        if i+1 < len(I):
            Fill(img, mag, high, low, i+1, j)
        if i-1 >= 0:
            Fill(img, mag, high, low, i-1, j)
        if i+1 < len(I) and j+1 < len(I[0]):
            Fill(img, mag, high, low, i+1, j+1)
        if i+1 < len(I) and j-1 >=0:
            Fill(img, mag, high, low, i+1, j-1)
        if j+1 < len(I[0]):
            Fill(img, mag, high, low, i, j+1)
        if j-1 >=0:
            Fill(img, mag, high, low, i, j-1)
        if j+1 < len(I[0]) and i-1 >=0:
            Fill(img, mag, high, low, i-1, j+1)
        if j-1 >=0 and i-1 >=0:
            Fill(img, mag, high, low, i-1, j-1)
    return

def Supression(xr, yr, Ipx, Ipy, M, NM):
    '''
    Non-maximum supression on image M
    For all pixels in M:
        Calculate slope at that pixel
        Identify direction of gradient
            tan(22.5) = 0.4142
            tan(67.5) = 2.4142
        Only include pixels that are greater than both its neighbors in the direction of the gradient.
    '''
    for x in xr:
        for y in yr:
            if Ipx[x][y] == 0:
                # account for divide by zero
                Ipx[x][y] = 0.00001 
            slope = Ipy[x][y] / Ipx[x][y] 
            if x < len(I) and x > 0 and y < len(I[0]) and y > 0:
                if slope <= 0.4142 and slope > -0.4142:
                    if M[x][y] > M[x][y-1] and M[x][y] > M[x][y+1]:
                        NM[x][y] = M[x][y]
                elif slope <= 2.4142 and slope > 0.4142:
                    if M[x][y] > M[x-1][y-1] and M[x][y] > M[x-1][y+1]:
                        NM[x][y] = M[x][y]
                elif slope <= -0.4142 and slope > -2.4142:
                    if M[x][y] > M[x+1][y-1] and M[x][y] > M[x-1][y+1]:
                        NM[x][y] = M[x][y]
                elif M[x][y] > M[x-1][y] and M[x][y] > M[x+1][y]:
                    NM[x][y] = M[x][y]
    return NM

# Read a gray scale image you can find from Berkeley Segmentation Dataset, Training images, store it as a matrix named I.
image = Image.open('image.jpg')
I = np.asarray(image)
fig=plt.figure()

# Create a one-dimensional Gaussian mask G to convolve with I. 
# The standard deviation(s) of this Gaussian is a parameter to the edge detector (call it σ > 0).
sigma = 0.5
G = np.asarray([GaussFilter(4, sigma)])
GT = np.transpose(G)

# Create a one-dimensional mask for the first derivative of the Gaussian in the x and y directions; call these Gx and Gy. 
# The same σ > 0 value is used as in step 2.
Gx = np.asarray([[1,0,-1]])
Gy = np.transpose(Gx)

# Convolve the image I with G along the rows to give the x component image (Ix), 
# and down the columns to give the y component image (Iy).
Ix = convolve(I, G, mode='same')
Iy = convolve(I, GT, mode='same')

# Plot Ix and Iy
fig.add_subplot(2, 3, 1)
plt.imshow(Ix, cmap='gray')
fig.add_subplot(2, 3, 2)
plt.imshow(Iy, cmap='gray')

# Convolve Ix with Gx to give I'x, the x component of I convolved with the derivative of the Gaussian, 
# and convolve Iy with Gy to give I'y, y component of I convolved with the derivative of the Gaussian.
Ipx = convolve(Ix, Gx, mode='same',method='fft')
Ipy = convolve(Iy, Gy, mode='same',method='fft')

# Plot I'x and I'y
fig.add_subplot(2, 3, 3)
plt.imshow(Ipx, cmap='gray')
fig.add_subplot(2, 3, 4)
plt.imshow(Ipy, cmap='gray')

# Compute the magnitude of the edge response by combining the x and y components. 
yr = range(0, (len(I[0])-1))
xr = range(0, (len(I)-1))
M = np.zeros((len(I),len(I[0])))

# Compute magnitude at each pixel
for x in xr:
    for y in yr:
        M[x][y] = np.sqrt(Ipx[x][y]**2 + Ipy[x][y]**2)

# Plot Magnitude 
fig.add_subplot(2, 3, 5)
plt.imshow(M, cmap='gray')

# Implement non-maximum suppression algorithm that we discussed in the lecture. 
# Pixels that are not local maxima should be removed with this method. 
# In other words, not all the pixels indicating strong magnitude are edges in fact. 
# We need to remove false-positive edge locations from the image.
NM = np.zeros((len(I), len(I[0])))
NM = Supression(xr, yr, Ipx, Ipy, M, NM)

# Apply Hysteresis thresholding to obtain final edge-map. You may use any existing library function to compute connected components if you want.

# Normalize image
img = np.zeros((len(I), len(I[0])))
maximum = np.ndarray.max(NM)
norm = NM / maximum
high = 0.35
low = 0.1

# Create Hysteresis thresholding mask
for x in xr:
    for y in yr:
        if(norm[x][y] > high):
            Fill(img, norm, high, low, x, y)

# Only include pixels from non-maximum supression that are marked as
# true in the thresholding mask
for x in xr:
    for y in yr:
        if img[x][y] == 1:
            img[x][y] = NM[x][y]

# Plot Output
fig.add_subplot(2, 3, 6)
plt.imshow(img, cmap='gray')

plt.show()