#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Rasti Najim
# DATE CREATED: 17 May 2019
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir
from PIL import Image


# Imports classifier function for using CNN to classify images
# from classifier import classifier
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below


def main():
    # TODO: 0. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # TODO: 1. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    

    # TODO: 2. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)
    

    # TODO: 3. Define classify_images() function to create the classifier
    # labels with the classifier function using in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    
    # TODO: 4. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)
    

    # TODO: 5. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)
    
    check_classifying_labels_as_dogs(result_dic)
    
    # TODO: 6. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch)

    # TODO: 0. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 0. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int(tot_time/3600)) + ':' +
          str(int(tot_time % 3600 / 60)) + ':' + str(int((tot_time % 3600) % 60)))


# TODO: 2.-to-7. Define all the function below. Notice that the input
# parameters and return values have been left in the function's docstrings.
# This is to provide guidance for achieving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to achieve the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the folder of pet images')

    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model architecture to use for image classification')

    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='Text file that contains all labels associated to dogs')

    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these labels as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    petlabel_dic = dict()
    filename_list = listdir(image_dir)

    for idx in range(0, len(filename_list)):
        if filename_list[idx][0] != '.':
            pet_label = ''
            pet_name = filename_list[idx].split('_')
            
            for word in pet_name:
                if word.isalpha():
                    pet_label += word + ' '
                    
                    petlabel_dic[filename_list[idx]
                                 ] = pet_label.lower().strip()

    return petlabel_dic


def classify_images(image_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()

    for key in petlabel_dic:
        model_label = classifier('{}/{}'.format(image_dir, key), model)
        model_label = model_label.lower().strip()

        truth = petlabel_dic[key]
        found = model_label.find(truth)

        if found >= 0:
            if ((found == 0 and len(truth) == len(model_label)) or
                    (((found == 0) or (model_label[found - 1] == " ")) and
                     ((found + len(truth) == len(model_label)) or
                      (model_label[found + len(truth): found+len(truth)+1] in
                               (",", " "))
                      )
                     )
                ):
                results_dic[key] = [truth, model_label, 1]
            else:
                results_dic[key] = [truth, model_label, 0]

        else:
            results_dic[key] = [truth, model_label, 0]

    return results_dic


def adjust_results4_isadog(result_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line.
                Dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dog_dic = dict()
    with open(dogsfile) as f:
        for line in f:
            line = line.rstrip()
            if line not in dog_dic:
                dog_dic[line] = 1

    

    for key in result_dic:
        if result_dic[key][0] in dog_dic:
            result_dic[key].append(1)

        else:
            result_dic[key].append(0)

        if result_dic[key][1] in dog_dic:
            result_dic[key].append(1)

        else:
            result_dic[key].append(0)


def calculates_results_stats(result_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    results_stats = dict()

    results_stats["n_images"] = len(result_dic)
    results_stats["n_dogs_img"] = 0
    results_stats["n_notdogs_img"] = 0
    results_stats["n_correct_dogs"] = 0
    results_stats["n_correct_notdogs"] = 0
    results_stats["n_correct_breed"] = 0
    results_stats["n_match"] = 0

    for key in result_dic:

        if result_dic[key][3] == 1 and result_dic[key][4] == 1:
            results_stats["n_correct_dogs"] += 1

        if result_dic[key][3] == 1:
            results_stats["n_dogs_img"] += 1

        if result_dic[key][3] == 0 and result_dic[key][4] == 0:
            results_stats["n_correct_notdogs"] += 1

        if result_dic[key][3] == 0:
            results_stats["n_notdogs_img"] += 1

        if result_dic[key][3] == 1 and result_dic[key][2] == 1:
            results_stats["n_correct_breed"] += 1

        if result_dic[key][2] == 1:
            results_stats["n_match"] += 1

    

       # % of correctly matched images
    results_stats["pct_match"] = results_stats["n_match"] / \
        results_stats["n_images"] * 100
       # % of correctly classified dog images
    results_stats["pct_correct_dogs"] = results_stats["n_correct_dogs"] / \
        results_stats["n_dogs_img"] * 100
       # % of correctly classified breed images
    results_stats["pct_correct_breed"] = results_stats["n_correct_breed"] / \
        results_stats["n_dogs_img"] * 100

       # % of correctly classified non_dog images
    if results_stats["n_notdogs_img"] > 0:
            results_stats["pct_correct_notdogs"] = results_stats["n_correct_notdogs"] / \
                results_stats["n_notdogs_img"] * 100

    else:
            results_stats["pct_correct_notdogs"] = 0.0

    return results_stats



def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """
    print('\nThe CNN model architecture that is being used: ' + model)

    print('\nNumber of images: ' + str(results_stats['n_images']))
    print('Number of Dog images: ' + str(results_stats['n_dogs_img']))
    print('Number of Not-a-Dog images: ' + str(results_stats['n_notdogs_img']))

    for key in results_stats:
        if key[0] == 'p':
            print('\n{}: {}'.format(key, results_stats[key]))

    if (print_incorrect_dogs==True and (results_stats['n_correct_dogs'] + results_stats['n_correct_notdogs'] != results_stats['n_images'])):

        print("\nINCORRECT Dog/NOT Dog Assignments:")

        for key in results_dic:
        
            if sum(results_dic[key][3:]) == 1:
                print('\nReal: {}        Classifier: {}'.format(results_dic[key][0], results_dic[key[1]]))

    if (print_incorrect_breed == True and (results_stats['n_correct_dogs'] != results_stats['n_correct_breed'])):

        print("\nINCORRECT Dog Breed Assignment:")

        for key in results_dic:

            if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0):
                print('Real: {}        Classifier: {}'.format(
                results_dic[key][0], results_dic[key[1]]))






# Call to main function to run the program
if __name__ == "__main__":
    main()
