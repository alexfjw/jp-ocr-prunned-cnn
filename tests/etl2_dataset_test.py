from src.datasets import Etl2Dataset
from utils.utf8 import print_utf8
import sys

print('testing etl2')
etl2 = Etl2Dataset()

# check out of bounds
etl2[len(etl2) - 1]

# print a character
first_image, first_label = etl2[0]
print_utf8(first_label)

# display the image
first_image.show()

# display the range
print('extrema:', first_image.getextrema())

# count num categories
character_set = set()
for character_entry in etl2:
    character_set.add(character_entry[1])

print(len(character_set))

# print mean
print('mean: ', etl2.mean)

# print std
print('std: ', etl2.std())
