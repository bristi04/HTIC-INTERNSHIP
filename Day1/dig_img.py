import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

nature_img=cv2.imread('nature.jpeg')

#RGB
nature_img_rgb=cv2.cvtColor(nature_img, cv2.COLOR_BGR2RGB)

#GrayScale
img_gs=cv2.cvtColor(nature_img_rgb, cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(5, 2))

plt.subplot(1, 3, 1)
plt.imshow(nature_img_rgb)
plt.title("RGB Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_gs, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

#Histogram of the Image

x,y=img_gs.shape
pix_freq=np.zeros(256, dtype=int)
for i in range(256):
    count=0
    for j in range(x):
        for k in range(y):
            if img_gs[j,k]==i:
                count+=1
    pix_freq[i]=count

n=np.arange(256)
plt.subplot(1, 3, 3) 
plt.stem(n,pix_freq)
plt.grid(True)
plt.ylabel('Frequency')
plt.xlabel('Intensity Levels')
plt.title('Histogram of the image')

plt.tight_layout()
plt.show()
