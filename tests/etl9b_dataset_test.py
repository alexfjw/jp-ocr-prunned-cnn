from src.datasets import Etl9bDataset
from utils.utf8 import print_utf8
import sys

print('testing etl9b')
etl9b = Etl9bDataset()

# check out of bounds
etl9b[len(etl9b) - 1]

# print a character
first_image, first_label = etl9b[0]
print_utf8(first_label)

# display the image
first_image.show()

# display the range, should be 0-255
print('extrema:', first_image.getextrema())

# count num categories
character_set = set()
for character_entry in etl9b:
    character_set.add(character_entry[1])

print(len(character_set))

# print mean
# print('mean: ', etl9b.mean)

# print std
# print('std: ', etl9b.std())
