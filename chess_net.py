import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn

import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np


"""
import warnings
warnings.filterwarnings("ignore")
"""


class ChessDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        #self.board_state = data["board_state"]
        #self.board_eval = data["board_eval"]
        self.board_state = data["b"]
        self.board_eval = data["v"]
        self.board_eval = np.asarray(self.board_eval / abs(self.board_eval).max() / 2 + 0.5, dtype=np.float32)

    def __len__(self):
        return self.board_state.shape[0]

    def __getitem__(self, idx):
        return (self.board_state[idx], self.board_eval[idx])


#linear layer = dense layer
# 14 separate channels? one for each matrix/tensor?
class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            #nn.Conv2d(8?, 16, kernel_size=3, padding=1),
            nn.Conv2d(14, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
        )

        self.last = nn.Linear(128, 1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(-1, 128)
        x = self.last(x)

        return torch.tanh(x)


#def train(net, dataset/path, epoch, criterion, optimizer, scheduler /decay?, lr=0.001):
#def train(net: nn.Module, path, number_of_samples, nepochs, lr=0.001):
#def train(path, saveas, nepochs, lr=0.003):
def train(model, dataset_path, nepochs=20, lr=0.003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #dataset = ChessDataset()
    train_set = ChessDataset(dataset_path)

    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # aka: criterion
    lossfunction = nn.MSELoss()

    model = model.to(device)
    model.train()

    for epoch in range(nepochs):
        total_loss = 0
        curr_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            label = label.unsqueeze(-1)
            data = data.to(device)
            label = label.to(device)

            data = data.float()
            label = label.float()

            optimizer.zero_grad()
            output = model(data)

            loss = lossfunction(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            curr_loss += 1
        print("%3d: %f" % (epoch, total_loss/curr_loss))



