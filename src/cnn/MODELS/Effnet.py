from torch import nn
from efficientnet_pytorch import EfficientNet


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        #self.densenet121 = torchvision.models.densenet121(pretrained=True)
        self.densenet121 = EfficientNet.from_pretrained('efficientnet-b5')
        #num_ftrs = self.densenet121.classifier.in_features
        num_ftrs = self.densenet121._fc.in_features
        
        self.densenet121._fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x