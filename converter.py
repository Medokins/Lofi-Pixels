from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import os

input_image = Image.open(os.path.join("data", "to_convert", "test.jpg"))

input_array = np.array(input_image)
pixels = input_array.reshape(-1, 3)

k = 6  # needs more testing
kmeans = KMeans(n_clusters=k, random_state=33).fit(pixels)

labels = kmeans.labels_
labels = labels.reshape(input_array.shape[:2])

output_array = np.zeros_like(input_array)
for i in range(k):
    output_array[labels == i] = kmeans.cluster_centers_[i]

output_image = Image.fromarray(output_array)
output_image.save(os.path.join("data", "converted", "converted_test.jpg"))
