from os import linesep
import sys
import random
import numpy as np

random.seed(1337)

# Split the datasets to 80% training and 20% testing randomly, only include unique passwords with 8-12 in length.
"""
change data path
"""
with open("../data/phpbb.txt", "r", encoding="utf-8", errors="ignore") as f:

    lines = f.readlines()

    print("{} passwords are in the dataset.".format(len(lines)))
    # lines = list(set(lines))
    # print("{} are unique".format(len(lines)))
    lines_12 = [x for x in lines if len(x) >= 8 and len(x) <= 12]
    print("{} are in range of 8 to 12".format(len(lines_12) / len(lines)))
    # print('[info] shuffling')
    # random.shuffle(lines_12)

    # split = int(len(lines_12) * 0.80)

    # change path for storing splited training and testing data

    with open("../data/phpbb812.txt", "w") as f:
        print("Savded")
        f.write("".join(lines_12[:]))

    # with open('../data/test_weakpass812.txt', 'w') as f:
    #     print('[info] saving 20% ({}) of dataset for testing in ../data/test_weakpass812.txt'.format(len(lines_12) - split))
    #     f.write(''.join(lines_12[split:]))
