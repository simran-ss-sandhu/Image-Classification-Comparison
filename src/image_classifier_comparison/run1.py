import os
import math
import imageio.v3 as iio
import numpy as np
import PIL.Image as Pillow
import json
import logging
from image_classifier_comparison.constants import CATEGORIES, TRAINING_DATA_FOLDER_PATH, TESTING_DATA_FOLDER_PATH


TRAINING_VECTORS_FOLDER_PATH = os.path.join("data", "processed", "run1", "training")
TINY_IMG_SIZE = 16
TRAINING_IMG_PER_CATEGORY = 80
SIZE_SQUARED = int(math.pow(TINY_IMG_SIZE, 2))
K = 49
RESULTS_OUTPUT_FILE_PATH = os.path.join("outputs", "predictions", "run1.txt")


# Convert an image to a vector
def __extract_tiny_image_feature(image) -> np.ndarray:
    image.astype(dtype=float)
    result = __crop_to_square(image)
    result = __downscale(result)
    result = __concatenate(result)
    result = __zero_mean(result)
    result = __unit_length(result)
    return result


# Crops an image to a square, about the center
def __crop_to_square(img: np.ndarray) -> np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    w_diff = max(width - height, 0)
    h_diff = max(height - width, 0)
    half_w_diff = w_diff / 2.0
    half_h_diff = h_diff / 2.0
    # Set bounds
    left = math.floor(half_w_diff)
    right = width - (math.ceil(half_w_diff))
    up = math.floor(half_h_diff)
    down = height - (math.ceil(half_h_diff))
    # Crop
    new_img = img[up:down, left:right]
    return new_img


# Downscales image to 16x16
def __downscale(img) -> np.ndarray:
    global TINY_IMG_SIZE
    pillow_img = Pillow.fromarray(img)
    pillow_img = pillow_img.resize((TINY_IMG_SIZE, TINY_IMG_SIZE))
    return np.array(pillow_img)


# Flattens 2D array to 1D array
def __concatenate(arr: np.ndarray) -> np.ndarray:
    arr = arr.flatten()
    return arr


# Sets mean of array to 0
def __zero_mean(arr: np.ndarray) -> np.ndarray:
    # Get current mean
    mean = np.mean(arr)
    # Subtract mean from each value
    new_arr = np.subtract(arr, mean)
    return new_arr


# Converts vector to unit vector
def __unit_length(arr: np.ndarray):
    length = np.linalg.norm(arr)
    arr = arr / length
    return arr


# Imports all vectors of training images, with categories
def __import_vectors() -> np.ndarray:
    global SIZE_SQUARED, TRAINING_IMG_PER_CATEGORY, TRAINING_VECTORS_FOLDER_PATH
    all_categories = np.zeros((len(CATEGORIES), TRAINING_IMG_PER_CATEGORY, SIZE_SQUARED), dtype=float)
    for i in range(0, len(CATEGORIES)):
        with (open(os.path.join(TRAINING_VECTORS_FOLDER_PATH, CATEGORIES[i] + '.txt'), 'r') as file):
            for e in range(0, TRAINING_IMG_PER_CATEGORY):
                arr_text = file.readline()
                all_categories[i][e] = np.array(json.loads(arr_text[:-1]))
    return all_categories


# Use k-nearest neighbour to match image to the closest category
def __categorise(img: np.ndarray, print_neighbours: bool, all_vecs: np.ndarray, k: int) -> str:
    global TRAINING_IMG_PER_CATEGORY
    distances = []
    # Get all distances
    for x in range(0, len(CATEGORIES)):
        category = CATEGORIES[x]
        for y in range(0, TRAINING_IMG_PER_CATEGORY):
            distances.append((int(__vector_distance(__extract_tiny_image_feature(img), all_vecs[x][y])), category))
    # Sort distances
    distances.sort(key=lambda d: d[0])
    tops = distances[:k]
    cats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, k):
        cats[CATEGORIES.index(tops[i][1])] += 1
    highest = 0
    highest_index = 0
    # Find most common category
    for e in range(0, len(CATEGORIES)):
        if cats[e] > highest:
            highest = cats[e]
            highest_index = e
    if print_neighbours:
        for e in range(0, len(CATEGORIES)):
            if cats[e] != 0:
                logging.info(CATEGORIES[e], ": ", cats[e])
    return CATEGORIES[highest_index]


# Calculates vector distance between two points
def __vector_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    global SIZE_SQUARED
    power_sum = 0
    for i in range(0, SIZE_SQUARED):
        power_sum += abs(math.pow(v1[i] - v2[i], 2))
    return math.sqrt(power_sum)


def extract_training_data_features():
    logging.info("extracting features from training data for run1...")

    # create required folders if they don't exist
    os.makedirs(TRAINING_VECTORS_FOLDER_PATH, exist_ok=True)

    for category in CATEGORIES:
        file_path = os.path.join(TRAINING_VECTORS_FOLDER_PATH, category + '.txt')
        with open(file_path, 'w') as file:
            for i in range(0, TRAINING_IMG_PER_CATEGORY):

                image_path = os.path.join(TRAINING_DATA_FOLDER_PATH, category, str(i) + ".jpg")
                image = iio.imread(image_path)

                vector = __extract_tiny_image_feature(image)

                # store extracted features
                file.write(str(vector.tolist()))
                file.write('\n')

    logging.info("training data for run1 processed")


def __test_run1():
    img_file = input("File URI: ")
    image = iio.imread(img_file)
    all_vecs = __import_vectors()
    result = __categorise(image, True, all_vecs, K)
    logging.info(result)
    return result


def __do_run1(img: np.ndarray) -> str:
    all_vecs = __import_vectors()
    result = __categorise(img, False, all_vecs, K)
    return result


# Test the accuracy with a specific k-value for k-nearest neighbour
def __test_k_accuracy(k):
    score = 0
    scores = [0] * len(CATEGORIES)
    max_score = (100 - TRAINING_IMG_PER_CATEGORY) * len(CATEGORIES)
    all_vecs = __import_vectors()
    for category_i in range(0, len(CATEGORIES)):
        category = CATEGORIES[category_i]
        for j in range(TRAINING_IMG_PER_CATEGORY, 100):
            image = iio.imread(os.path.join(TRAINING_DATA_FOLDER_PATH, category, str(j) + ".jpg"))
            result = __categorise(image, False, all_vecs, k)
            if result == category:
                score += 1
                scores[category_i] += 1
        logging.info(f"loading {int((category_i / len(CATEGORIES)) * 100)}%")
    accuracy = score
    logging.info(f"Accuracy all: {(score / max_score) * 100}%")
    logging.info("===")
    for j in range(0, len(CATEGORIES)):
        logging.info(f"Accuracy {CATEGORIES[j]}: {(scores[j] / (max_score / len(CATEGORIES))) * 100}%")
    return accuracy


def tune_k_value():
    logging.info("finding the best k-value for k-nearest neighbour")
    highest_score = 0
    best_k = 0
    for k in range(1, 100):
        logging.info(k)
        score = __test_k_accuracy(k)
        if score > highest_score:
            highest_score = score
            best_k = k
    logging.info(f"Best k-value = {best_k}")


def categorise_testing_data():
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_FILE_PATH), exist_ok=True)
    with open(RESULTS_OUTPUT_FILE_PATH, 'w') as file:
        for i in range(0, 2988):
            logging.info(i)
            if os.path.isfile(os.path.join(TESTING_DATA_FOLDER_PATH, str(i) + ".jpg")):
                image = iio.imread(os.path.join(TESTING_DATA_FOLDER_PATH, str(i) + ".jpg"))
                file.write(str(i) + ".jpg " + str.lower(str(__do_run1(image))))
                file.write('\n')
