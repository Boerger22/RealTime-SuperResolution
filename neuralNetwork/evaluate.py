import dataset
import keras
import metrics
import os
import tensorflow as tf
import yaml

from pathlib import Path


def evaluate(model_path: str, config_path: str, evaluation_path: str = None):
    # load model
    model = load_keras_model(model_path)

    # check whether a config is given such that a test_set according to the config can be used. Otherwise, a default test_set has to be created
    if config_path == None:
        # try to search for config file in model path
        config_path = find_config(model_path)

        test_dataset = create_test_set(config_path)

        if evaluation_path == None:
            evaluation_path = model_path.split("keras_model/")[0].split("save/")[1][:-1]
    else:
        print("The model could not be evaluated because neither a data set nor a configuration was given.")
        exit()

    if len(test_dataset.hr_images) > 0:  # only test on dataset if size > 0
        evaluate_model(model, test_dataset, evaluation_path)


def evaluate_model(model: keras.Model, test_dataset: dataset, model_path: str = None, evaluation_path: str = None):
    print("Evaluating Model.")

    result = model.evaluate(test_dataset)
    evaluation = dict(zip(model.metrics_names, result))

    # save evaluation to model folder if no evaluation path is given
    if evaluation_path == None:
        evaluation_path = "save/" + model_path + "/evaluation/" if model_path != None else "./"

    # prepare folder
    Path(evaluation_path).mkdir(parents=True, exist_ok=True)

    # save evaluation
    with open(evaluation_path + "evaluation.yaml", "w") as evaluation_file:
        yaml.dump(evaluation, evaluation_file, default_flow_style=False)
    print("Evaluation finished.")


def load_keras_model(model_path: str):
    return keras.models.load_model(model_path, custom_objects={"psnr": metrics.psnr, "ssim": metrics.ssim})


def find_config(model_path):
    print("Searching for config.yaml in model_path.")
    for root, dirs, files in os.walk(model_path.split("keras_models/")[0]):
        if "config.yaml" in files:
            print("Found a config.yaml in {}/".format(root))
            return root + "/"
    return ""


def create_test_set(config_path: str = ""):
    if config_path != "":
        test_dataset = dataset.Dataset(set_type="test", path=config_path, seeding=True)
    else:
        print("Couldnt find config. Using the default configuration file.")
        test_dataset = dataset.Dataset(set_type="test", seeding=True)

    return test_dataset
