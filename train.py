import torch
from chess_net import CNN, train

"""
import warnings
warnings.filterwarnings("ignore")
"""


def train_chessnet(net_to_train, save_path, c, d):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #net/model
    model = CNN()
    model = model.to(device)
    model.load_state_dict(torch.load(net_to_train))
    train(model, datasets)
    torch.save(model.state_dict(), save_path)




if __name__ == "__main__":
    train_chessnet('a','b','c','d')