import os
import argparse
import numpy as np
import pickle


'''
python3 accuracy.py --input-generated generated/gnpassgan/10/180000iter/8.txt --input-test data/test_rockyou10.txt
'''

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-generated', '-g',
                        dest='input_generated',
                        help='The generated text file')

    parser.add_argument('--input-test', '-t',
                        dest='input_test',
                        help='The test file for comparison')
    
    parser.add_argument('--input-duplicates', '-d',
                        dest='input_duplicates',
                        help='The file containing passwords in both test and generated file')

    return parser.parse_args()

args = parse_args()


def accuracy(path_generated, path_test):
    generated_file = open(path_generated, 'r').readlines()
    test_data = open(path_test, 'r').readlines()
    generated_data = set(generated_file)
    test_data = set(test_data)
    print("The number of unique password generated is {}.". format(len(generated_data)))
    print("The number of unique password in the test set is {}.". format(len(test_data)))
    overlap = generated_data & test_data
    print("{} passwords are correctly guessed.".format(len(overlap)) )
    result = round(float(len(overlap))/len(test_data) * 100, 4)
    return print('The guessing accuracy is {} %'.format(result))

accuracy(args.input_generated, args.input_test)

'''
Command line to generate a file for all duplicates: comm -12 <(sort generated/gnpassgan/200000iter/generated9.txt) <(sort data/test_rockyou10.txt) > dups.txt
'''

# # for larger dataset (10 to the power of 9 or more)
# def accuracy(path_generated, path_test):
#     count = 0
#     overlap = []
#     generated_file = open(path_generated, 'r')
#     test_data = open(path_test, 'r').readlines()
#     print("These passwords are correctly guessed:")
#     while True:
#         count += 1
#         # Get next line from file
#         line = generated_file.readline()
#         if line in test_data:
#             overlap.append(line)
#             print("Line{}: {}".format(len(overlap), line.strip()))
#         # if line is empty
#         # end of file is reached
#         if not line:
#             break
#     generated_file.close()
#     print("{} passwords are correctly guessed.".format(len(overlap)) )
#     result = round(float(len(overlap))/len(test_data) * 100, 4)
#     return print('The guessing accuracy is {} %'.format(result))
# accuracy(args.input_generated, args.input_test)
    