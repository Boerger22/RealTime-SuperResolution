import train
import yaml


def train_schedule():
    activation_functions = ["hard_sigmoid", "hard_sigmoid_modified"]

    with open("./config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    for i, activation_function in enumerate(activation_functions):
        config["activation_function"] = activation_function

        with open("./config.yaml", "w") as output_file:
            yaml.dump(config, output_file, default_flow_style=False)
        train(str(i))


if __name__ == "__main__":
    train_schedule()
