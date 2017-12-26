from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from functools import reduce
import bitstring
import codecs
from PIL import Image

# misc code for extracting data from files
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

    def __init__(self):
        # files to item count
        self.files = [('data/ETL2/ETL2_1', 9056),
                      ('data/ETL2/ETL2_2', 10480),
                      ('data/ETL2/ETL2_3', 11360),
                      ('data/ETL2/ETL2_4', 10480),
                      ('data/ETL2/ETL2_5', 11420)
                      ]
        self.entries = []
        self.save_entries_to_memory()

    def save_entries_to_memory(self):
        image_size = (60, 60)
        bits_per_pixel = 6
        for file_directory, item_count in self.files:
            file = bitstring.ConstBitStream(filename=file_directory)

            # loop through the items in each file
            for item_index in range(item_count):

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
                pil_image = Image.frombytes('F', image_size, item_data[16], 'bit', bits_per_pixel)

                self.entries.append(
                    CharacterEntry(pil_image=pil_image,
                                   label=label)
                )

    def __len__(self):
        def sum_file_count(sum_so_far, file_with_count):
            return sum_so_far + file_with_count[1]

        return reduce(sum_file_count, self.files, 0)

    def __getitem__(self, idx):
        return self.entries[idx]

# def etl9bDataset(Dataset):