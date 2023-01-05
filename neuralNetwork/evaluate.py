import datasets.dataset as dataset
import keras
import os
import tensorflow as tf
import utils.metrics as metrics
import yaml

from pathlib import Path


def evaluate(model_path: str, seeding: bool, config_path: str = None, evaluation_path: str = None):
    # load model
    model = load_keras_model(model_path)

    # check whether a config is given such that a test_set according to the config can be used. Otherwise, a default test_set has to be created
    test_dataset = create_test_set(seeding, config_path)

    if evaluation_path == None:
        evaluation_path = "save/" + model_path.split("keras_model/")[0].split("save/")[1][:-1] + "/evaluation/"

    if len(test_dataset.hr_images) > 0:  # only test on dataset if size > 0
        evaluate_model(model, test_dataset, evaluation_path)


def evaluate_model(model: keras.Model, test_dataset: dataset, evaluation_path: str):
    print("Evaluating Model.")

    result = model.evaluate(test_dataset)
    evaluation = dict(zip(model.metrics_names, result))

    # prepare folder
    Path(evaluation_path).mkdir(parents=True, exist_ok=True)

    # save evaluation
    with open(evaluation_path + "evaluation.yaml", "w") as evaluation_file:
        yaml.dump(evaluation, evaluation_file, default_flow_style=False)
    print("Evaluation finished.")


def load_keras_model(model_path: str):
    return keras.models.load_model(model_path, custom_objects={"psnr": metrics.psnr, "ssim": metrics.ssim})


def create_test_set(seeding, config_path: str = ""):
    if config_path != None:
        test_dataset = dataset.Dataset(set_type="test", path=config_path, seeding=seeding)
    else:
        print("Couldnt find config. Using the default configuration file.")
        test_dataset = dataset.Dataset(set_type="test", seeding=seeding)

    return test_dataset
