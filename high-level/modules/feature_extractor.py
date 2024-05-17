import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class DepthFeatureExtractor(nn.Module):
    def __init__(self, input_dim=5, output_dim=64):
        super(DepthFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 80x80

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 20x20

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 10x10

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 5x5
        )

        # Flattening the tensor and passing through Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.output_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0)


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, latent_dim, output_activation=None, num_channel=1):
        super().__init__()

        self.num_frames = num_channel
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, latent_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class DepthOnlyFCBackbone54x96(nn.Module):
    def __init__(self, latent_dim, output_activation=None, num_channel=1):
        super().__init__()

        self.num_frames = num_channel
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 54, 96]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 50, 92]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 25, 46]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 23, 44]
            activation,
            nn.Flatten(),
            # Calculate the flattened size: 64 * 23 * 44
            nn.Linear(64 * 23 * 44, 128),
            activation,
            nn.Linear(128, latent_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent




class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channel, feature_vector_length):
        super(CNNFeatureExtractor, self).__init__()
        
        # Define the first convolutional block
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, padding=1)
        # Use a smaller stride in the pooling layer to reduce downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Define the second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Optionally, you can skip a pooling layer or use a larger kernel size
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Calculate the size of the flattened feature maps
        # After two rounds of pooling with different strides, the size will be reduced less
        # We use one pooling with stride 1 and one with stride 2
        self.flattened_size = 128 * (150 - 1 - 1//2) * (120 - 1 - 1//2)
        
        # Define the fully connected layer
        self.fc1 = nn.Linear(self.flattened_size, 256)
        
        # Define the output layer
        self.features = nn.Linear(256, feature_vector_length)

    def forward(self, x):
        # Apply the first convolutional block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply the second convolutional block
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Apply the third convolutional block
        x = F.relu(self.conv3(x))  # Skipping pooling here
        
        # Flatten the feature maps
        x = x.view(-1, self.flattened_size)
        
        # Apply the fully connected layer
        x = F.relu(self.fc1(x))
        
        # Output the feature vector
        x = self.features(x)
        
        return x
