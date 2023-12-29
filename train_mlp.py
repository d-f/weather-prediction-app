import utils
import multi_layer_perceptron
import argparse


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_classes")
    parser.add_argument("-input_size")
    parser.add_argument("-hidden_size")
    return parser.parse_args()


def get_model(num_classes, input_size, hidden_size):
    return multi_layer_perceptron.MLP(
        num_classes=num_classes, input_size=input_size, hidden_size=hidden_size
    )


def get_dataset()


def main():
    args = create_argparser()
    model = get_model(num_classes=args.num_classes, input_size=args.input_size, hidden_size=args.hidden_size)



if __name__ == "__main__":
    main()
