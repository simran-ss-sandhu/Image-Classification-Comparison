import os
import pandas as pd
import numpy as np
import sklearn
import shutil
import concurrent.futures
import logging
import image_classifier_comparison.GIST as GIST
from image_classifier_comparison.constants import TRAINING_DATA_FOLDER_PATH, TESTING_DATA_FOLDER_PATH

TRAINING_FEATURES_AND_LABELS_FILE_PATH = os.path.join("data", "processed", "run3", 'training_features_and_labels.csv')
TESTING_FEATURES_FILE_PATH = os.path.join("data", "processed", "run3", 'testing_features.csv')
RESULTS_OUTPUT_FILE_PATH = os.path.join("outputs", "predictions", "run3.txt")
BEST_SVM_PARAMS = {'C': 2.0, 'kernel': 'rbf', 'gamma': 'auto'}


def __create_training_dataset_process_folder(folder_path: str):
    """Calculate the gist descriptor of every image in a folder
    :param folder_path: path of the folder
    :return: gist descriptor of every image in the folder and their associated label
    """

    label = os.path.basename(folder_path)

    logging.info(f"Started {label}...")
    image_paths = [
        os.path.join(folder_path, imageName)
        for imageName in os.listdir(folder_path)
        if imageName.lower().endswith(('.jpg', '.jpeg'))
    ]
    gists = GIST.calculate_gist_descriptors(image_paths)
    logging.info(f"...done {label}")

    return gists, [label] * len(gists)


def create_training_dataset():
    """Generate the gist descriptor of every image in the training set and store it in a csv file with its label"""

    logging.info("Creating training dataset...")

    data = []
    targets = []

    # extract all gist descriptors and targets
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # gets all folder paths in the training directory
        folder_paths = []
        for filename in os.listdir(TRAINING_DATA_FOLDER_PATH):
            folder_path = os.path.join(TRAINING_DATA_FOLDER_PATH, filename)
            if os.path.isdir(folder_path):
                folder_paths.append(folder_path)

        # split folder processing across cores
        results = [
            executor.submit(__create_training_dataset_process_folder, folder_path)
            for folder_path in folder_paths]

        # wait for all results
        concurrent.futures.wait(results)

        # combines results
        for result in results:
            gists, labels = result.result()
            data.extend(gists)
            targets.extend(labels)

    logging.info('...extracted features of all images')

    # convert to numpy array
    data = np.array(data)
    targets = np.array(targets)

    # add features to dataframe
    num_of_features = data.shape[1]
    df = pd.DataFrame(data, columns=['Feature ' + str(i) for i in range(num_of_features)])
    logging.info('...stored data in dataframe')

    # add targets to dataframe
    df['targets'] = targets
    logging.info('...stored targets in dataframe')

    # store in a file for later use
    os.makedirs(os.path.dirname(TRAINING_FEATURES_AND_LABELS_FILE_PATH), exist_ok=True)
    df.to_csv(TRAINING_FEATURES_AND_LABELS_FILE_PATH, index=False)
    logging.info('...stored dataset in a file')


def __create_testing_dataset_process_images(image_paths: list[str]):
    """Calculate the gist descriptor of every image in a group of images
    :param image_paths: paths of every image in the group
    :return: gist descriptor of every image in the group and their filename
    """

    logging.info("Group started...")
    descriptors = GIST.calculate_gist_descriptors(image_paths)
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    logging.info("...done group")

    return descriptors, image_names


def create_testing_dataset():
    """Generate the gist descriptor of every image in the testing set and store it in a csv file with its image name"""

    logging.info("Creating testing dataset...")

    descriptors = []
    image_names = []

    with concurrent.futures.ProcessPoolExecutor() as executor:

        image_paths = [
            os.path.join(TESTING_DATA_FOLDER_PATH, filename)
            for filename in os.listdir(TESTING_DATA_FOLDER_PATH)
            if filename.lower().endswith(('.jpg', '.jpeg'))
        ]

        # split image paths into groups (of 100) for multiprocessing
        group_size = 100
        grouped_image_paths = [image_paths[i:i + group_size] for i in range(0, len(image_paths), group_size)]

        # send each group of image paths to each core
        results = [executor.submit(__create_testing_dataset_process_images, group) for group in grouped_image_paths]

        # wait for all results
        concurrent.futures.wait(results)

        # combine results
        for result in results:
            ds, image_name = result.result()
            descriptors.extend(ds)
            image_names.extend(image_name)

    logging.info('...extracted features of all images')

    # convert from python list to numpy array
    descriptors = np.array(descriptors)

    # add features to dataframe
    num_of_features = descriptors.shape[1]
    df = pd.DataFrame(descriptors, columns=['Feature ' + str(i) for i in range(num_of_features)])
    logging.info('...stored data in dataframe')

    # add image name to dataframe
    df['names'] = image_names
    logging.info('...stored image names in dataframe')

    # store in a file
    os.makedirs(os.path.dirname(TESTING_FEATURES_FILE_PATH), exist_ok=True)
    df.to_csv(TESTING_FEATURES_FILE_PATH, index=False)
    logging.info('...stored dataset in a file')


def find_best_svm_params():
    """Find the best parameters for the SVM classifier"""

    logging.info("Finding best SVM classifier parameters...")

    # parameters to search through
    model_param_grid = {
        'C': np.arange(0.1, 100, 0.1),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['auto', 'scale', 0.1, 1, 10]
    }

    # load the dataset
    df = pd.read_csv(TRAINING_FEATURES_AND_LABELS_FILE_PATH)
    x = np.array(df.drop('targets', axis=1))
    y = np.array(df.targets)

    # standardise the features
    scaler = sklearn.preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    # find best parameters
    svm = sklearn.svm.SVC(random_state=69)
    grid_search = sklearn.model_selection.GridSearchCV(
        estimator=svm,
        param_grid=model_param_grid,
        cv=10,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(x, y)
    logging.info("...found best parameters")
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_}")


def test_model():
    """Use 10-fold cross validation to test the model on the training data"""

    logging.info("Testing model...")

    # load the dataset
    df = pd.read_csv(TRAINING_FEATURES_AND_LABELS_FILE_PATH)
    x = np.array(df.drop('targets', axis=1))
    y = np.array(df.targets)

    accuracies = []

    kf = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
    for train_index, test_index in kf.split(x, y):

        # split into training and testing sets
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # standardise the features
        scaler = sklearn.preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # train SVM classifier
        svm = sklearn.svm.SVC(
            kernel=BEST_SVM_PARAMS['kernel'],
            C=BEST_SVM_PARAMS['C'],
            gamma=BEST_SVM_PARAMS['gamma'],
            random_state=69)
        svm.fit(x_train, y_train)
        y_hat = svm.predict(x_test)

        # calculate accuracy
        accuracy = sklearn.metrics.accuracy_score(y_test, y_hat)
        accuracies.append(accuracy)

    logging.info("...done")

    # print test results
    logging.info(f'Average accuracy: {np.mean(accuracies)}')
    logging.info(f'Highest accuracy: {np.max(accuracies)}')
    logging.info(f'Lowest accuracy: {np.min(accuracies)}')


def predict_testing_dataset():
    """
    Predict the testing dataset an RBF classifier trained with the training dataset.
    The predictions are written to 'run3.txt'
    """

    logging.info("Predicting classes for the testing dataset...")

    # load the datasets
    df_train = pd.read_csv(TRAINING_FEATURES_AND_LABELS_FILE_PATH)
    x_train = np.array(df_train.drop('targets', axis=1))
    y_train = np.array(df_train.targets)
    df_test = pd.read_csv(TESTING_FEATURES_FILE_PATH)
    x_test = np.array(df_test.drop('names', axis=1))

    # standardise the features
    scaler = sklearn.preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # train RBF model
    svm = sklearn.svm.SVC(
        kernel=BEST_SVM_PARAMS['kernel'],
        C=BEST_SVM_PARAMS['C'],
        gamma=BEST_SVM_PARAMS['gamma'],
        random_state=69)
    svm.fit(x_train, y_train)

    # predict classes for test data
    y_pred = svm.predict(x_test)
    logging.info("...done")

    # write predictions to file
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_FILE_PATH), exist_ok=True)
    with open(RESULTS_OUTPUT_FILE_PATH, 'w') as file:
        for path, prediction in zip(df_test.names, y_pred):
            image_name = os.path.basename(path)
            file.write(image_name + ' ' + prediction + '\n')
    logging.info("...written predictions to 'run3.txt'")


def create_predicted_testing_directory():
    """
    Create a folder directory organised according to the predictions in 'run3.txt'
    :return:
    """

    logging.info("Creating directory according to testing predictions...")

    # make root directory
    shared_directory = os.path.join("data", "processed", "run3", 'predicted_testing')
    os.makedirs(shared_directory, exist_ok=True)

    # copy images to the folder associated with their predicted class
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_FILE_PATH), exist_ok=True)
    with open(RESULTS_OUTPUT_FILE_PATH, 'r') as file:
        for line in file:

            image_name, assigned_class = line.strip().split()

            # create class directory if it doesn't exist
            class_directory = os.path.join(shared_directory, assigned_class)
            os.makedirs(class_directory, exist_ok=True)

            # copy image
            shutil.copy2(os.path.join(TESTING_DATA_FOLDER_PATH, image_name), class_directory)

    logging.info("...done")
