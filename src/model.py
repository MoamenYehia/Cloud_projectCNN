import torch 
import torch.nn as nn 

class CNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            #Block 1 
            nn.Conv2d(in_channels=3 , out_channels=32 , kernel_size=3 , padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            #Block 2
            nn.Conv2d(in_channels=32 , out_channels=64 , kernel_size=3 , padding =1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
        )

        self.classifier = nn.Linear(in_features=64*16*16 , out_features=num_classes)

    def forward(self , x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    model=CNNModel(num_classes=7)
    test_tensor=torch.randn(1,3,64,64)
    output=model(test_tensor)
    print("Verfication for the model working right:", output.shape)  # Expected output shape: (1, 7)

