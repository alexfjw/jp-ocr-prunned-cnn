from collections import namedtuple
from torch.utils.data import Dataset
from functools import reduce
import bitstring
import codecs
import struct
import pickle
import skimage.filters as filters
import numpy as np
from PIL import Image, ImageStat

# code for extracting data from files
# refer to http://etlcdb.db.aist.go.jp/?page_id=1721
t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'


def T56(c):
    return t56s[c]


with codecs.open('data/co59-utf8.txt', 'r', 'utf-8') as co59f:
    co59t = co59f.read()

co59l = co59t.split()
CO59 = {}
for c in co59l:
    ch = c.split(':')
    co = ch[1].split(',')
    CO59[(int(co[0]), int(co[1]))] = ch[0]

CharacterEntry = namedtuple('CharacterEntry', ['pil_image', 'label'])


class Etl2Dataset(Dataset):
    """
    images without transform are in greyscale
    mean & std are single channel, and calculated in advance
    """
    mean = 43.87300058377132
    std = 11.3157826035
    # manually determined by adding into a set, see etl2_dataset_test.py

    def __init__(self, train_transforms=None, test_transforms=None):
        # files to item count
        self.files = [('data/ETL2/ETL2_1', 9056),
                      ('data/ETL2/ETL2_2', 10480),
                      ('data/ETL2/ETL2_3', 11360),
                      ('data/ETL2/ETL2_4', 10480),
                      ('data/ETL2/ETL2_5', 11420)
                      ]
        self.train = True
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.entries = self.load_entries_to_memory()
        self.classes, self.class_to_idx = self.load_class_data()

    def load_class_data(self):
        classes = list({label for _, label in self.entries})
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def load_entries_to_memory(self):
        file_name = 'data/etl2_entries.obj'
        try:
            file_handler = open(file_name, 'rb')
            entries = pickle.load(file_handler)
            print('restored pickled etl2 data')
            return entries

        except:
            print('processing raw etl2 data')
            entries = []
            image_size = (60, 60)
            bits_per_pixel = 6
            for file_directory, num_items in self.files:
                file = bitstring.ConstBitStream(filename=file_directory)

                # loop through the items in each file
                for item_index in range(num_items):

                    file.pos = item_index * 6 * 3660
                    item_data = file.readlist('int:36,uint:6,pad:30,6*uint:6,6*uint:6,pad:24,2*uint:6,pad:180,bytes:2700')

                    # specifications about each item's data
                    # http://etlcdb.db.aist.go.jp/?page_id=1721
                    # 0 -> serial index
                    # 1 -> source, in T56
                    # 2:8 -> name of type of character, kanji, kana, in T56
                    # 8:14 -> name of font type, in T56
                    # 14:16 -> label
                    # 16 -> image bits
                    # print item_data[0], T56(r[1]), "".join(map(T56, item_data[2:8])), "".join(map(T56, r[8:14])), CO59[tuple(r[14:16])])

                    # save only the label & image
                    label = CO59[tuple(item_data[14:16])]

                    # image is grayscale, use otsu's algorithm to binarize it
                    pil_image = Image.frombytes('F', image_size, item_data[16], 'bit', bits_per_pixel)
                    np_image = np.array(pil_image)
                    global_threshold = filters.threshold_otsu(np_image)
                    binarized_image = np_image > global_threshold
                    # fromarray '1' is buggy, convert array to 0 & 255 uint8,
                    # then build image with PIL as 'L' & convert to '1'
                    pil_image = Image.fromarray((binarized_image * 255).astype(np.uint8), mode='L')

                    entries.append(
                        CharacterEntry(pil_image=pil_image,
                                       label=label)
                    )

            # save the data to file so we don't have to load it again
            file_handler = open(file_name, 'wb')
            pickle.dump(entries, file_handler)
            return entries

    def __len__(self):
        def sum_file_count(sum_so_far, file_with_count):
            return sum_so_far + file_with_count[1]

        return reduce(sum_file_count, self.files, 0)

    def __getitem__(self, idx):
        label = self.entries[idx].label
        image = self.entries[idx].pil_image

        if self.train and self.train_transforms:
            image = self.train_transforms(image)
        elif not self.train and self.test_transforms:
            image = self.test_transforms(image)

        return image, self.class_to_idx[label]


class Etl9bDataset(Dataset):
    """
    images without transform are in greyscale
    mean & std are single channel, and calculated in advance
    """
    mean = 63.243677227077974
    std = 16.8771143038
    num_classes = 3036

    def __init__(self, train_transforms=None, test_transforms=None):
        # files to item count
        self.files = [('data/ETL9B/ETL9B_1', 121440),
                      ('data/ETL9B/ETL9B_2', 121440),
                      ('data/ETL9B/ETL9B_3', 121440),
                      ('data/ETL9B/ETL9B_4', 121440),
                      ('data/ETL9B/ETL9B_5', 121440+3036)
                      ]
        self.entries = []
        self.train = True
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.load_entries_to_memory()
        # TODO: pickle me

    def load_entries_to_memory(self):
        image_size = (64, 63)
        record_size = 576

        for file_directory, num_items in self.files:

            with open(file_directory, mode='rb') as f:
                for item_index in range(num_items):
                    # 0 is a dummy item, shift all by 1
                    f.seek((item_index+1) * record_size)
                    s = f.read(record_size)
                    r = struct.unpack('>2H4s504s64x', s)
                    # refer to https://stackoverflow.com/questions/25134722/convert-shift-jis-to-utf-8
                    # to find out how to get unicode from the shiftjis hexcode
                    # iso2022 is has an escape sequence prefix, '\x1b$B'
                    label = (b'\x1b$B' + bytes.fromhex(hex(r[1])[2:])).decode('iso2022_jp')
                    pil_image = Image.frombytes('1', image_size, r[3], 'raw')

                    self.entries.append(
                        CharacterEntry(pil_image=pil_image,
                                       label=label)
                    )

    # image_stats = ImageStat.Stat(pil_image)
    # item_count = item_index + 1
    # # cumulative moving average
    # self.mean = self.mean + (image_stats.mean[0] - self.mean)/item_count
    # self.mean2 = self.mean2 + (np.square(image_stats.mean[0]) - self.mean2)/item_count


    def __len__(self):
        def sum_file_count(sum_so_far, file_with_count):
            return sum_so_far + file_with_count[1]

        return reduce(sum_file_count, self.files, 0)

    def __getitem__(self, idx):
        label = self.entries[idx].label
        image = self.entries[idx].pil_image

        if self.train and self.train_transforms:
            image = self.train_transforms(image)
        elif not self.train and self.test_transforms:
            image = self.test_transforms(image)

        return image, label
