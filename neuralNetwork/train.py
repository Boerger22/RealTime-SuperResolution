import dataset
import keras
import metrics
import numpy as np
import random as python_random
import tensorflow as tf
import yaml

from datetime import datetime
from model import FSRCNN_s_PReLU
from pathlib import Path
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


tf.keras.mixed_precision.set_global_policy('mixed_float16')


def train(config_path: str = "./config.yaml", prefix: str = ""):
    # start timer
    time_start = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    # read in config
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    seed = config["seed"]

    # set seeds
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)

    # define model
    gpu_string = "GPU" if config["tf_gpu"] else "CPU"
    model = FSRCNN_s_PReLU()
    model_name = model.getModelName() + "_" + gpu_string + "_" + str(config["max_number_images"]) + "images" + "_" + str(
        config["lr_size"][1]) + "_" + str(config["lr_size"][0]) + "_x" + str(config["model_upscaling_factor"]) + "_" + str(config["epochs"]) + "ep" + prefix
    model = model.getModel()

    # compile model
    model.compile(
        optimizer=config["model_optimizer"],
        loss=config["model_loss"],
        metrics=['accuracy', metrics.psnr, metrics.ssim])

    # prepare directories and files
    weights_save_directory, history_path, save_config_path, evaluation_path = prepareDirectories(config, model_name)

    # prepare datasets
    train_dataset, val_dataset, test_dataset = prepareDatasets(config)

    # wrap datasets
    datasets = [train_dataset, val_dataset, test_dataset]

    # define callbacks
    callbacks = []
    # save only best weights of the model
    weights_save = keras.callbacks.ModelCheckpoint(
        filepath=weights_save_directory + "/model_{epoch:05d}.h5",
        monitor="loss",
        save_best_only=True,
        save_weights_only=True
    )
    # save the history in a CSV
    history_logger = keras.callbacks.CSVLogger(
        history_path + "history.csv", separator=";", append=False)

    logdir = "save/" + model_name + "/fit_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    callbacks = [weights_save, history_logger, tensorboard_callback]

    # training
    print("Started training of ", model_name)

    # get training device
    tf_device = "/GPU:0" if config["tf_gpu"] else "/CPU:0"
    with tf.device(tf_device):
        model.fit(
            train_dataset,
            epochs=config["epochs"],
            callbacks=callbacks,
            validation_data=val_dataset,
        )
    print("Training for {} done!".format(model_name))

    evaluate_model(model, config, test_dataset, evaluation_path)

    save_model(model, model_name, config, save_config_path, datasets, callbacks)

    # end timer
    time_end = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    with open("save/" + model_name + "/evaluation/time_logger.txt", "w") as time_file:
        time_file.write("Started: {}, finished: {}".format(time_start, time_end))


def prepareDirectories(config: dict, model_name: str):
    print("Preparing directories and files")
    # Determine directory for weights saving, history.csv and config
    weights_save_directory = "save/" + model_name + "/" + "weights/"
    history_path = "save/" + model_name + "/history/"
    save_config_path = "save/" + model_name + "/config/"
    evaluation_path = "save/" + model_name + "/evaluation/"

    # Check if paths and files exist
    Path(weights_save_directory).mkdir(parents=True, exist_ok=True)
    Path(history_path).mkdir(parents=True, exist_ok=True)
    Path(history_path + "/history.csv").touch(exist_ok=True)
    Path(save_config_path).mkdir(parents=True, exist_ok=True)
    Path(evaluation_path).mkdir(parents=True, exist_ok=True)
    if(config["save_training_images"]):
        Path("data/debugImages/LR/train/").mkdir(parents=True, exist_ok=True)
        Path("data/debugImages/LR/val/").mkdir(parents=True, exist_ok=True)
        Path("data/debugImages/LR/test/").mkdir(parents=True, exist_ok=True)
        Path("data/debugImages/HR/train/").mkdir(parents=True, exist_ok=True)
        Path("data/debugImages/HR/val/").mkdir(parents=True, exist_ok=True)
        Path("data/debugImages/HR/test/").mkdir(parents=True, exist_ok=True)

    return weights_save_directory, history_path, save_config_path, evaluation_path


def prepareDatasets(config: dict):
    print("Preparing datasets")

    # Prepare training-, validation-, and test-dataset
    train_dataset = dataset.Dataset(set_type="train")
    val_dataset = dataset.Dataset(set_type="val")
    test_dataset = dataset.Dataset(set_type="test")

    train_dataset_size = len(train_dataset.hr_images)
    val_dataset_size = len(val_dataset.hr_images)
    test_dataset_size = len(test_dataset.hr_images)
    print("Batch size: ", config["batch_size"])
    print("Size of training set: ", train_dataset_size)
    print("Size of validation set: ", val_dataset_size)
    print("Size of test set: ", test_dataset_size)

    print("Preparing datasets done!")

    return train_dataset, val_dataset, test_dataset


def evaluate_model(model: keras.Model, config: dict, test_dataset: dataset, evaluation_path: str):
    if(config["test_split"] != 0):
        result = model.evaluate(test_dataset)
        evaluation = dict(zip(model.metrics_names, result))

    # Evaluation
    with open(evaluation_path + "evaluation.yaml", "w") as evaluation_file:
        yaml.dump(evaluation, evaluation_file, default_flow_style=False)


def save_model(model: keras.Model, model_name: str, config: dict, save_config_path: str, datasets: list, callbacks):
    # Saving
    print("Saving {} model...".format(model_name))

    # Frozen Graph
    x = tf.TensorSpec(model.input_shape, model.inputs[0].dtype, name="input")

    concrete_function = tf.function(lambda x: model(x)).get_concrete_function(x)
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    tf.io.write_graph(frozen_model.graph, "save/" + model_name + "/frozen_graph/", "model.pb", as_text=False)

    # keras model
    model.save("save/" + model_name + "/keras_models/")

    # save model config
    config["train_dataset_size"] = len(datasets[0].hr_images)
    config["val_dataset_size"] = len(datasets[1].hr_images)
    config["test_dataset_size"] = len(datasets[2].hr_images)
    config["callbacks"] = list(map(lambda callback: str(callback).split(" object")[0].replace("<", ""), callbacks))
    with open(save_config_path + "config.yaml", "w") as output_file:
        yaml.dump(config, output_file, default_flow_style=False)

    print("Saving {} model done!".format(model_name))


if __name__ == "__main__":
    train()
