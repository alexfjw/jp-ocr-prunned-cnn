from collections import namedtuple
from torch.utils.data import Dataset
from sklearn import preprocessing
from functools import reduce
import bitstring
import codecs
import struct
import pickle
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageStat

CharacterEntry = namedtuple('CharacterEntry', ['pil_image', 'label'])

class Etl2Dataset(Dataset):
    """
    images without transform are in greyscale
    mean & std are single channel, and calculated in advance
    """
    mean = 29.9074215166
    std = 65.5108579121
    # manually determined by adding into a set, see etl2_dataset_test.py

    # code for extracting data from files
    # refer to http://etlcdb.db.aist.go.jp/?page_id=1721
    t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'

    def T56(c):
        return Etl2Dataset.t56s[c]

    with codecs.open('raw_data/co59-utf8.txt', 'r', 'utf-8') as co59f:
        co59t = co59f.read()

    co59l = co59t.split()
    CO59 = {}
    for c in co59l:
        ch = c.split(':')
        co = ch[1].split(',')
        CO59[(int(co[0]), int(co[1]))] = ch[0]

    def __init__(self, train_transforms=None, test_transforms=None):
        # files to item count
        self.files = [('raw_data/ETL2/ETL2_1', 9056),
                      ('raw_data/ETL2/ETL2_2', 10480),
                      ('raw_data/ETL2/ETL2_3', 11360),
                      ('raw_data/ETL2/ETL2_4', 10480),
                      ('raw_data/ETL2/ETL2_5', 11420)
                      ]
        self.train = True
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self._entries = self.load_entries_to_memory()
        self.label_encoder = self.load_class_data()

    def load_class_data(self):
        le = preprocessing.LabelEncoder()
        le.fit([label for _, label in self._entries])
        return le

    def load_entries_to_memory(self):
        file_name = 'raw_data/etl2_entries.obj'
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
                    label = Etl2Dataset.CO59[tuple(item_data[14:16])]

                    # image is grayscale, use otsu's algorithm to binarize it
                    pil_image = Image.frombytes('F', image_size, item_data[16], 'bit', bits_per_pixel)
                    # np_image = np.array(pil_image)
                    # global_threshold = filters.threshold_otsu(np_image)
                    # binarized_image = np_image > global_threshold
                    # fromarray '1' is buggy, convert array to 0 & 255 uint8,
                    # then build image with PIL as 'L' & convert to '1'
                    # pil_image = Image.fromarray((binarized_image * 255).astype(np.uint8), mode='L')

                    entries.append(
                        CharacterEntry(pil_image=pil_image,
                                       label=label)
                    )

            # save the data to file so we don't have to load it again
            file_handler = open(file_name, 'wb')
            pickle.dump(entries, file_handler)
            return entries

    def calculate_mean(self):
        return np.mean([ImageStat.Stat(image).mean[0] for image, _ in self._entries])

    def calculate_std(self):
        return np.mean([ImageStat.Stat(image).stddev[0] for image, _ in self._entries])

    def __len__(self):
        def sum_file_count(sum_so_far, file_with_count):
            return sum_so_far + file_with_count[1]

        return reduce(sum_file_count, self.files, 0)

    def __getitem__(self, idx):
        label = self._entries[idx].label
        image = self._entries[idx].pil_image.convert('I')

        if self.train and self.train_transforms:
            image = self.train_transforms(image)
        elif not self.train and self.test_transforms:
            image = self.test_transforms(image)

        return image, self.label_encoder.transform([label])[0]


class Etl9GDataset(Dataset):
    """
    images without transform are in greyscale
    mean & std are single channel, and calculated in advance
    """
    mean = 11.3169761515
    std = 39.1004076498

    def __init__(self, train_transforms=None, test_transforms=None):
        # files to item count
        self.files_directory = 'raw_data/ETL9G'
        self.train = True
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self._entries = self.load_entries_to_memory()
        self.label_encoder = self.load_class_data()

    def load_class_data(self):
        le = preprocessing.LabelEncoder()
        le.fit([label for _, label in self._entries])
        return le

    def load_entries_to_memory(self):
        print('processing raw etl9g data')
        # don't bother pickling, size is about 4gb, and takes about as long as reading each file

        image_size = (128, 127)
        record_size = 8199
        items_per_file = 12144
        entries = []

        for file_path in [path for path in Path(self.files_directory).iterdir()]:
            with open(file_path, mode='rb') as f:
                for item_index in range(items_per_file):
                    # 0 is a dummy item, shift all by 1
                    f.seek(item_index * record_size)
                    s = f.read(record_size)
                    r = struct.unpack('>2H8sI4B4H2B34x8128s7x', s)
                    # refer to https://stackoverflow.com/questions/25134722/convert-shift-jis-to-utf-8
                    # to find out how to get unicode from the shiftjis hexcode
                    # iso2022 is has an escape sequence prefix, '\x1b$B'
                    label = (b'\x1b$B' + bytes.fromhex(hex(r[1])[2:])).decode('iso2022_jp')
                    pil_image = Image.frombytes('F', image_size, r[14], 'bit', 4)

                    entries.append(
                        CharacterEntry(pil_image=pil_image,
                                       label=label)
                    )

        return entries

    def calculate_mean(self):
        return np.mean([ImageStat.Stat(image).mean[0] for image, _ in self._entries])

    def calculate_std(self):
        return np.mean([ImageStat.Stat(image).stddev[0] for image, _ in self._entries])

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        label = self._entries[idx].label
        image = self._entries[idx].pil_image.convert('I')

        if self.train and self.train_transforms:
            image = self.train_transforms(image)
        elif not self.train and self.test_transforms:
            image = self.test_transforms(image)

        return image, self.label_encoder.transform([label])[0]


class Etl_2_9G_Dataset(Dataset):

    def __init__(self, etl2_transforms=None, etl9g_transforms=None):
        self.etl2 = Etl2Dataset(etl2_transforms['train'], etl2_transforms['test'])
        self.etl9g = Etl9GDataset(etl9g_transforms['train'], etl9g_transforms['test'])
        self.label_encoder = self.load_class_data()

    def load_class_data(self):
        classes = list(set(self.etl2.label_encoder.classes_)
                       | set(self.etl9g.label_encoder.classes_))

        le = preprocessing.LabelEncoder()
        le.fit(classes)
        return le

    def __len__(self):
        return len(self.etl2) + len(self.etl9g)

    def __getitem__(self, idx):
        if idx < len(self.etl2):
            img, old_label_idx = self.etl2[idx]
            label = self.etl2.label_encoder.inverse_transform([old_label_idx])[0]
            return img, self.label_encoder.transform([label])[0]
        else:
            img, old_label_idx = self.etl9g[idx - len(self.etl2)]
            label = self.etl9g.label_encoder.inverse_transform([old_label_idx])[0]
            return img, self.label_encoder.transform([label])[0]
