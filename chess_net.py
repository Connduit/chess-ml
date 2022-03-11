import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn

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
        """
        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)
        """

        self.last = nn.Linear(128, 1)

        #self.conv1 = nn.Conv2d() 
        #self.fc1 = nn.Linear(2048,64)
        #self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(-1, 128)
        x = self.last(x)
        """
        #print(x.size())
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)
        """

        return torch.tanh(x)




#def train(net, dataset/path, epoch, criterion, optimizer, scheduler /decay?, lr=0.001):
#def train(net: nn.Module, path, number_of_samples, nepochs, lr=0.001):
def train(path, saveas, nepochs, lr=0.003):
    #torch.manual_seed(var)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #dataset = ChessDataset()
    #TODO: below is wrong? it needs to be a ChessDataset()
    #train_set = train_set_gen.get_dataset(num_samples=number_of_samples)
    #path = "processed/dataset_2k.npz"
    train_set = ChessDataset(path)

    #train_data = 
    #train_loader = DataLoader(train_set, batch_size=30, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # aka: criterion
    lossfunction = nn.MSELoss()

    model = model.to(device)
    model.train()



    for epoch in range(nepochs):
        # do loss stuff here?
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
        #torch.save(model.state_dict(), "nets/my_values.pth")
        torch.save(model.state_dict(), saveas)





if __name__ == "__main__":
    train("processed/my_dataset25k.npz", 10)