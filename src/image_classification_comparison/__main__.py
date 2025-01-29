import logging
import image_classification_comparison.run1 as run1
import image_classification_comparison.run2 as run2
import image_classification_comparison.run3 as run3

logging.basicConfig(level=logging.INFO, format="|%(asctime)s|%(name)s|%(levelname)s| %(message)s")


def main():

    # ==RUN1==
    run1.extract_training_data_features()
    # run1.tune_k_value()
    run1.categorise_testing_data()

    # ==RUN2==
    run2.train_model()
    run2.categorise_testing_data()

    # ==RUN3==
    run3.create_training_dataset()
    # run3.find_best_svm_params()  # TAKES AGES TO DO
    run3.test_model()
    run3.create_testing_dataset()
    run3.predict_testing_dataset()
    # run3.create_predicted_testing_directory()

    pass


if __name__ == "__main__":
    main()
