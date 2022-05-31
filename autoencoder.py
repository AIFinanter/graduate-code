import torch.nn as nn

class lowtohigh(nn.Module):
    def __init__(self,h_height,h_width,scale=4):
        super(lowtohigh,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*h_height*h_width,3*(h_height//2)*(h_width//2)),
            nn.ReLU(),
            nn.Linear(3*(h_height//2)*(h_width//2),3*(h_height//4)*(h_width//4)),
            nn.ReLU(),
        )

    def forward(self,x):
        x = self.encoder(x)
        return x


