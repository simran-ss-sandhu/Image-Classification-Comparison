import os
import numpy as np
import random
import joblib
import imageio.v3 as iio
import json
import sklearn
import sklearn.cluster
import scipy
import logging
from image_classifier_comparison.constants import CATEGORIES, TRAINING_DATA_FOLDER_PATH, TESTING_DATA_FOLDER_PATH


PATCH_SIZE = 8
STRIDE_LENGTH = 4

SAMPLES = 200000
CLUSTERS = 2000

TRAINED_MODEL_FOLDER_PATH = os.path.join("data", "processed", "run2", "training")
TRAINING_IMG_PER_CATEGORY = 80
RESULTS_OUTPUT_FILE_PATH = os.path.join("outputs", "predictions", "run2.txt")


# returns list of feature vectors, given an image
def __extract_patches(img):
    patches = []
    for y in range(0, img.shape[0] - PATCH_SIZE + 1, STRIDE_LENGTH):
        for x in range(0, img.shape[1] - PATCH_SIZE + 1, STRIDE_LENGTH):
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if not np.std(patch) == 0:
                patch = (patch - np.mean(patch)) / np.std(patch)
            else:
                patch = patch - np.mean(patch)
            patches.append(patch.flatten())
    return np.array(patches)


# randomly selects vectors from list of vectors, given number to select
def __select(number_of_samples, vectors):
    selected_vectors = []
    indexes = list(range(len(vectors)))
    random.shuffle(indexes)
    for x in range(number_of_samples):
        selected_vectors.append(vectors[indexes[x]])
    return np.array(selected_vectors)


# returns k-means model given vectors and number of clusters
def __k_means(patches, n_clusters):
    kmeans = sklearn.cluster.KMeans(n_clusters)
    kmeans.fit_predict(patches)
    return kmeans


# returns list of histograms of word counts given list of quantised vectors, per category
def __create_arrays(category_vectors, labels, n_clusters):
    histogram_list = np.zeros((15, n_clusters))
    start_index = 0
    for x in range(0, 15):
        another_list = category_vectors[x]
        for y in range(start_index, start_index + len(another_list)):
            if labels[y] != -1:
                histogram_list[x][labels[y]] += 1
        start_index += len(another_list)
    return histogram_list


# returns histogram of word counts given list of quantised vectors
def __one_array(labels, n_clusters):
    histogram = np.zeros(n_clusters)
    for x in range(len(labels)):
        if labels[x] != -1:
            histogram[labels[x]] += 1
    return histogram


# Finds closest matching category histogram to img histogram, to return closest category
def __classify(histogram, histogram_list):
    high_score = 0
    highest_category = -1
    for x in range(len(histogram_list)):
        similarity = 1 - scipy.spatial.distance.cosine(histogram, histogram_list[x])
        if similarity > high_score:
            high_score = similarity
            highest_category = x
    return highest_category


# remap a value from one range to another
# based on py5.remap
# values outside the initial range are not clipped
def __remap(value, in_min, in_max, out_min, out_max):
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def __get_inverse_frequencies(histogram_list: np.ndarray, n_clusters: int):
    means = np.zeros(n_clusters)
    highest_mean = 0

    # calculate the sum for each cluster
    for arr in histogram_list:
        for i in range(0, n_clusters):
            means[i] += arr[i]
            if means[i] > highest_mean:
                highest_mean = means[i]

    # normalise the means
    means /= n_clusters
    highest_mean /= n_clusters

    # apply remapping and transform
    for a in range(0, n_clusters):
        means[a] = (1 - __remap(means[a], 0, highest_mean, 0, 1)) ** 10

    # adjust the histograms
    adjusted_histograms = np.ndarray(np.shape(histogram_list))
    for e in range(0, 15):
        adjusted_histograms[e] = histogram_list[e] * means

    return means, adjusted_histograms


def train_model():
    logging.info("training model for run2...")

    vectors = []  # features per category
    joined_vectors = []  # features for all categories

    for category in CATEGORIES:
        logging.info(f"extracting features from {category} category...")

        category_vectors = []  # features for the current category

        # find features for the current category
        for i in range(0, TRAINING_IMG_PER_CATEGORY):
            image = iio.imread(os.path.join(TRAINING_DATA_FOLDER_PATH, category, str(i) + ".jpg"))
            category_vectors += __extract_patches(image).tolist()
            logging.info(i)

        joined_vectors += category_vectors
        vectors.append(category_vectors)

        logging.info(f"...extracted features from {category} category")

    kmeans_model = __k_means(__select(SAMPLES, joined_vectors), CLUSTERS)

    # create required directories
    os.makedirs(TRAINED_MODEL_FOLDER_PATH, exist_ok=True)

    # save the kmeans model to a file
    joblib.dump(kmeans_model, os.path.join(TRAINED_MODEL_FOLDER_PATH, "kmeans_run2.joblib"))

    labels = kmeans_model.predict(joined_vectors)
    histograms = __create_arrays(vectors, labels, CLUSTERS)

    # save the histograms of the cluster labels for each category
    with open(os.path.join(TRAINED_MODEL_FOLDER_PATH, "Histograms.txt"), 'w') as file:
        file.write(str(histograms.tolist()))

    logging.info("...trained model for run2")


def __classify_image(img: np.ndarray) -> str:

    # load and process the histogram data
    with open(os.path.join(TRAINED_MODEL_FOLDER_PATH, "Histograms.txt"), 'r') as file:
        histogram_data = json.loads(file.readline())
    inverse_frequencies, adjusted_histograms = __get_inverse_frequencies(histogram_data, CLUSTERS)
    vectors = __extract_patches(img)

    # load the pretrained kmeans model
    kmeans_model = joblib.load(os.path.join(TRAINED_MODEL_FOLDER_PATH, "kmeans_run2.joblib"))

    labels = kmeans_model.predict(vectors)
    histogram = __one_array(labels, CLUSTERS)
    histogram *= inverse_frequencies
    category_index = __classify(histogram, adjusted_histograms)

    return CATEGORIES[category_index]


def calculate_training_data_accuracy():
    logging.info("calculating training data accuracy...")

    score = 0
    scores = [0] * len(CATEGORIES)
    max_score = (100 - TRAINING_IMG_PER_CATEGORY) * len(CATEGORIES)
    for category_i in range(0, len(CATEGORIES)):
        logging.info(f"category {category_i}/{len(CATEGORIES)}: {CATEGORIES[category_i]}")
        category = CATEGORIES[category_i]
        for i in range(TRAINING_IMG_PER_CATEGORY, 100):
            logging.info(i)
            image = iio.imread(os.path.join(TRAINING_DATA_FOLDER_PATH, category, str(i) + ".jpg"))
            result = __classify_image(image)
            if result == category:
                score += 1
                scores[category_i] += 1
    logging.info(f"Accuracy all: {(score / max_score) * 100}%")
    logging.info("===")
    for i in range(0, len(CATEGORIES)):
        logging.info(f"Accuracy {CATEGORIES[i]}: {(scores[i] / (max_score / len(CATEGORIES))) * 100}%")

    logging.info("...calculated training data accuracy")


def categorise_testing_data():
    logging.info("categorising testing data for run2...")
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_FILE_PATH), exist_ok=True)
    with open(RESULTS_OUTPUT_FILE_PATH, 'w') as file:
        for i in range(0, 2988):
            logging.info(i)
            if os.path.isfile(os.path.join(TESTING_DATA_FOLDER_PATH, str(i) + ".jpg")):
                image = iio.imread(os.path.join(TESTING_DATA_FOLDER_PATH, str(i) + ".jpg"))
                file.write(str(i) + ".jpg " + str.lower(str(__classify_image(image))))
                file.write('\n')
    logging.info("...categorised testing data for run2")
