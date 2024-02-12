#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: 
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images_solution.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Import necessary libraries
from time import time
from print_functions_for_lab_checks import *
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():
    # Measure total program runtime
    start_time = time()

    # Get command line arguments
    in_arg = get_input_args()

    # Check command line arguments
    check_command_line_arguments(in_arg)

    # Create results dictionary with pet labels
    results = get_pet_labels(in_arg.dir)
    check_creating_pet_image_labels(results)

    # Classify images using the specified CNN model architecture
    classify_images(in_arg.dir, results, in_arg.arch)
    check_classifying_images(results)

    # Adjust results to determine if the classifier correctly classified images as dogs or not
    adjust_results4_isadog(results, in_arg.dogfile)
    check_classifying_labels_as_dogs(results)

    # Calculate and collect results statistics
    results_stats = calculates_results_stats(results)
    check_calculating_results(results, results_stats)

    # Print summary results, incorrect classifications of dogs, and incorrectly classified breeds
    print_results(results, results_stats, in_arg.arch, True, True)

    # Measure total program runtime
    end_time = time()

    # Print overall runtime
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))


# Call to main function to run the program
if __name__ == "__main__":
    main()
