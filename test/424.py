from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

image = io.imread(r"C:\Users\dell\Pictures\Saved Pictures\t0173df002c22d721b4.jpg")
segments = slic(image, n_segments=150, compactness=10)
print(segments.shape)
n_liantong = segments.max() + 1
print("n_liantong: ", n_liantong)
area = np.bincount(segments.flat)
w, h = segments.shape
print(area / (w*h))
print((max(area/(w*h))), (min(area/(w*h))))

out = mark_boundaries(image, segments)
plt.subplot(111)
plt.imshow(out)
plt.show()
