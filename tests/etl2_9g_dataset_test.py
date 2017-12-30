from src.datasets import Etl_2_9G_Dataset
import numpy as np
from utils.utf8 import print_utf8

print('testing etl2_9g')
nones = {'train': None, 'test': None}
etl = Etl_2_9G_Dataset(nones, nones)

# check bounds
etl[len(etl) - 1]

# print a character
first_image, _ = etl[0]
print(np.array(first_image).shape)

# print a character
last_image, _ = etl[len(etl)-1]
print(np.array(last_image).shape)

# display the image
first_image.show()

# display the image
last_image.show()

# display the range, should be 0-255
print('extrema:', first_image.getextrema())

