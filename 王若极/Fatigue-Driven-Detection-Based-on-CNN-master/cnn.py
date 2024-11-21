from torch import nn 

class oneDcnn(nn.Module):
    def __init__(self, num_class=2): # 类别个数
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=8, stride=1),
            nn.ReLU(),

            nn.Conv1d(24, 24, kernel_size=4, stride=1),
            nn.ReLU(),

            nn.Conv1d(24, 24, kernel_size=4, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, num_class),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x