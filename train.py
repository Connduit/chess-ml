import torch
import numpy as np
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from chess_net import CNN

"""
import warnings
warnings.filterwarnings("ignore")
"""

class ChessDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        #self.X = data["arr_0"] = board states
        #self.Y = data["arr_1"] = game result
        self.board_state = data["board_state"]
        self.board_eval = data["board_eval"]
        pass

    def __len__(self):
        #return self.X.shape[0]
        return self.board_state.shape[0]

    def __getitem__(self, idx):
        return (self.board_state[idx], self.board_eval[idx])






if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    cds = ChessDataset()
    train_loader = DataLoader(cds, batch_size=256, shuffle=True)
    model = CNN()
    optimizer = optim.Adam(model.parameters())
    floss = nn.MSELoss()

    model = model.to(device=device)
    model.train()
    for epoch in range(10):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)
            data, target = data.to(device), target.to(device)

            data = data.float()
            target = target.float()


            optimizer.zero_grad()
            output = model(data)
            #print(output.shape)

            loss = floss(output, target)
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        print("%3d: %f" % (epoch, all_loss/num_loss))
        torch.save(model.state_dict(), "nets/value.pth")