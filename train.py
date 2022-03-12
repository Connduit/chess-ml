import argparse
import torch
from chess_net import CNN, train

"""
import warnings
warnings.filterwarnings("ignore")
"""

#data_path = "./processed"

def cli():
    arg_parser = argparse.ArgumentParser(description="Builds a Dataset from a .PGN File")
    arg_parser.add_argument("-m", "--model_path", required=True, help="Load Location for .PTH Model")
    arg_parser.add_argument("-s", "--save_model_path", required=True, help="Save Location for .PTH Model")
    arg_parser.add_argument("-d", "--dataset_path", required=True, help=".NPZ Dataset Location")
    #arg_parser.add_argument("-ns", "--num_samples", type=int, default=None, help="")
    #arg_parser.add_argument("-ng", "--num_games", type=int, default=None, help="")
    return arg_parser

def train_chessnet(model_path, save_model_path, dataset_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #net/model
    model = CNN()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    #train(model, dataset_path, save_model_path)
    train(model, dataset_path)
    torch.save(model.state_dict(), save_model_path)




if __name__ == "__main__":
    args = cli()
    model_path = vars(args.parse_args())["model_path"]
    save_path = vars(args.parse_args())["save_model_path"]
    dataset_path = vars(args.parse_args())["dataset_path"]
    train_chessnet(model_path, save_path, dataset_path)