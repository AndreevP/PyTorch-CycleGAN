import glob
from PIL import Image

import os
for pics_dir in ['datasets/horse2zebra/train/A', 'datasets/horse2zebra/train/B', 'datasets/horse2zebra/test/A', 'datasets/horse2zebra/test/B']:
    files_A = sorted(glob.glob(pics_dir + '/*.*'))
    for pic in files_A:
        img = Image.open(pic)
        if img.mode != 'RGB':
            os.remove(pic)
            print(pic, img.mode)
