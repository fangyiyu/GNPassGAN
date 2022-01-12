import os
import argparse
import numpy as np
import pickle



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


    
