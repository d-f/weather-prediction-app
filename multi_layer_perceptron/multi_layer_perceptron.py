import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, num_classes, input_size, hidden_size):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.layer1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer3 = nn.Linear(in_features=hidden_size*2, out_features=hidden_size*2)
        self.layer4 = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.layer5 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x))
