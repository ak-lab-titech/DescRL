
import torch
import torch.nn as nn
import torchvision.models as models


class VisionPredictor(nn.Module):
    def __init__(self, num_objects, num_regions):
        super(VisionPredictor, self).__init__()
        self.predictor = models.resnet18(pretrained=True)

        output_size = num_objects + num_regions
        num_ftrs = self.predictor.fc.in_features
        self.predictor.fc = nn.Linear(num_ftrs, output_size)  # modifying 1000 classes
        self.sigmoid = nn.Sigmoid()  # torch.sigmoid

    def forward(self, x):

        x = self.predictor(x)
        x = self.sigmoid(x)

        return x
