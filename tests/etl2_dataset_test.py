from src.datasets import Etl2Dataset
from utils.utf8 import print_utf8
import sys

print('testing etl2')
etl2 = Etl2Dataset()

# test len
etl2[len(etl2) - 1]

# test character
print_utf8(etl2[0].label)

#test image
etl2[0].pil_image.show()
