import yaml
from keras.layers import Conv2D, Input, PReLU, Conv2DTranspose, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')


def hard_sigmoid_modified(x):
    zero = tf.constant(0., dtype=x.dtype.base_dtype)
    one = tf.constant(1., dtype=x.dtype.base_dtype)

    x = tf.clip_by_value(x, zero, one)
    return x


get_custom_objects().update({'hard_sigmoid_modified': Activation(hard_sigmoid_modified)})


class FSRCNN_s_PReLU():
    def __init__(self):
        with open("config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        assert (config["lr_size"][0] * config["model_upscaling_factor"] == config["hr_size"][0]
                ), "The upscaling factor does not correspond to the given input and output size."
        assert (config["lr_size"][1] * config["model_upscaling_factor"] == config["hr_size"][1]
                ), "The upscaling factor does not correspond to the given input and output size."

        self.model_name = "FSRCNN"
        self.model = self.init_model(
            lr_size=config["lr_size"],
            feature_number=config["feature_number"],
            shrunk_feature_number=config["shrunk_feature_number"],
            color_channels=config["color_channels"],
            upscaling_factor=config["model_upscaling_factor"],
            activation_function=config["activation_function"])

    def init_model(
            self, lr_size: tuple, feature_number: int, shrunk_feature_number: int, color_channels: int,
            upscaling_factor: int, activation_function: str):
        lr_input = Input(shape=(lr_size[0], lr_size[1], color_channels))

        feature_extraction_conv = Conv2D(kernel_size=5, filters=feature_number, padding="same")(lr_input)

        feature_extraction_prelu = PReLU(shared_axes=[1, 2])(feature_extraction_conv)

        shrinking_conv = Conv2D(kernel_size=1, filters=shrunk_feature_number,
                                padding="same")(feature_extraction_prelu)

        shrinking_prelu = PReLU(shared_axes=[1, 2])(shrinking_conv)

        mapping_conv = Conv2D(kernel_size=3, filters=shrunk_feature_number, padding="same")(shrinking_prelu)

        mapping_prelu = PReLU(shared_axes=[1, 2])(mapping_conv)

        expanding_conv = Conv2D(kernel_size=1, filters=feature_number)(mapping_prelu)

        expanding_prelu = PReLU(shared_axes=[1, 2])(expanding_conv)

        deconvolution = Conv2DTranspose(
            kernel_size=9, filters=color_channels, strides=upscaling_factor, padding="same",
            activation=activation_function, dtype="float32")(expanding_prelu)

        return Model(inputs=lr_input, outputs=deconvolution)

    def getModel(self):
        return self.model

    def getModelName(self):
        return self.model_name
