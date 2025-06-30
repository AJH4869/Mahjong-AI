# Model part
import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(145, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 235)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_logits = self._tower(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetModel(nn.Module):
    def __init__(self, num_blocks=18):
        super(ResNetModel, self).__init__()
        self.conv1 = nn.Conv2d(145, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Create a sequential container with the specified number of ResidualBlocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_blocks)]
        )
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.fc2 = nn.Linear(256, 235)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        
        x = self.conv1(obs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.residual_blocks(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        action_logits = self.fc2(x)
        
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask


