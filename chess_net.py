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
        #self.board_state = data["arr_0"]
        #self.board_eval = data["arr_1"]

    def __len__(self):
        return self.board_state.shape[0]

    def __getitem__(self, idx):
        return (self.board_state[idx], self.board_eval[idx])


class ConvBlock(nn.Module):
    def __init__(self) -> None:
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73 # =4672 # needed?
        self.conv1 = nn.Conv2d(14, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
    def forward(self, x):
        x = x.view(-1, 14, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += res
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self) -> None:
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    def forward(self, x):
        v = F.relu(self.bn(self.conv(x))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v =  torch.tanh(self.fc2(v))
        #v = F.tanh(self.fc2(v))
        #TODO: i don't care about p? just the value head?
        p = F.relu(self.bn1(self.conv1(x))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v



class AlphaCNN(nn.Module):
    def __init__(self) -> None:
        super(AlphaCNN, self).__init__()
        self.conv = ConvBlock()
        # cuda out of memory so range has to be very low
        for block in range(1):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    def forward(self,s):
        s = self.conv(s)
        # cuda out of memory so range has to be very low
        for block in range(1):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


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


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error

#def train(net, dataset/path, epoch, criterion, optimizer, scheduler /decay?, lr=0.001):
#def train(net: nn.Module, path, number_of_samples, nepochs, lr=0.001):
def train(path, saveas, nepochs, lr=0.003):
    #torch.manual_seed(var)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #dataset = ChessDataset()
    train_set = ChessDataset(path)

    #train_data = 
    #train_loader = DataLoader(train_set, batch_size=30, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)

    model = CNN()
    #model = AlphaCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # aka: criterion
    lossfunction = nn.MSELoss()
    #lossfunction = AlphaLoss()

    model = model.to(device)
    model.train()



    for epoch in range(nepochs):
        # do loss stuff here?
        total_loss = 0
        curr_loss = 0
        #for idx, data in enumerate(train_loader):
        for idx, (data, label) in enumerate(train_loader):
            #s, v = data
            #v = v.unsqueeze(-1)
            label = label.unsqueeze(-1)
            data = data.to(device)
            label = label.to(device)

            data = data.float()
            label = label.float()
            #s = s.to(device).float()
            ##p = p.to(device).float()
            #v = v.to(device).float()

            optimizer.zero_grad()
            output = model(data)
            #s_pre, v_pre = model(s)

            loss = lossfunction(output, label)
            #loss = lossfunction(v_pre, v)
            ##loss = lossfunction(v_pre[:,0], v, p_pre, p)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            curr_loss += 1
        print("%3d: %f" % (epoch, total_loss/curr_loss))
        #torch.save(model.state_dict(), "nets/my_values.pth")
    torch.save(model.state_dict(), saveas)





if __name__ == "__main__":
    #train("processed/dataset_25k.npz", "nets/my_small25k.pth", 20)
    #train("dataset.npz", "nets/alpha1-5M.pth", 20)
    train("dataset.npz", "nets/alpha1-5M.pth", 20)