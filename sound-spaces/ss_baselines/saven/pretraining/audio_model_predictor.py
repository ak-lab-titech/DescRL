
import torch
import torch.nn as nn
import torchvision.models as models


class AudioPredictor(nn.Module):
    def __init__(self, num_objects, num_regions):
        super(AudioPredictor, self).__init__()
        self.num_objects = num_objects
        self.num_regions = num_regions

        self.predictor = models.resnet18(pretrained=True)
        self.predictor.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 2 channels in input for audio instead of 3 in image
        output_size = self.num_objects + self.num_regions
        num_ftrs = self.predictor.fc.in_features
        self.predictor.fc = nn.Linear(num_ftrs, output_size)  # modifying 1000 classes
        self.sigmoid = nn.Sigmoid()  # torch.sigmoid

    def forward(self, audio_observations):
        x = self.predictor(audio_observations)

        x1 = x[:, :self.num_objects]
        x2 = self.sigmoid(x[:, -self.num_regions:])
        x = torch.cat([x1, x2], dim=1)

        return x
