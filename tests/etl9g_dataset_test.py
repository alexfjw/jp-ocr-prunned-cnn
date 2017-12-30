from src.datasets import Etl9GDataset
from utils.utf8 import print_utf8

print('testing etl9g')
etl9g = Etl9GDataset()

# check bounds
etl9g[len(etl9g) - 1]

# print a character
first_image, _ = etl9g[0]

# display the image
first_image.show()

# display the range, should be 0-255
print('extrema:', first_image.getextrema())

# count num categories
character_set = set()
for character_entry in etl9g:
    character_set.add(character_entry[1])

print(len(character_set))

# print mean
print('mean: ', etl9g.calculate_mean())

# print std
print('std: ', etl9g.calculate_std())
