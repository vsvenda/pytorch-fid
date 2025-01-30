import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from keras.models import load_model
import numpy as np


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # First block
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv1_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second block
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv2_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv2_2.weight, mode='fan_in', nonlinearity='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third block
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv3_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv3_2.weight, mode='fan_in', nonlinearity='relu')
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv3_3.weight, mode='fan_in', nonlinearity='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth block
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv4_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv4_2.weight, mode='fan_in', nonlinearity='relu')
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv4_3.weight, mode='fan_in', nonlinearity='relu')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fifth block
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv5_1.weight, mode='fan_in', nonlinearity='relu')
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv5_2.weight, mode='fan_in', nonlinearity='relu')
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv5_3.weight, mode='fan_in', nonlinearity='relu')
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Applying layers
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Initialize model_1
        self.model1 = Model1()

        # First convolution block
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='linear')
        self.bn1 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.activation1 = nn.ReLU()  # ReLU activation
        self.dropout1 = nn.Dropout(0.0)  # Drop rate 0

        # Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolution block
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='linear')
        self.bn2 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.activation2 = nn.ReLU()  # ReLU activation
        self.dropout2 = nn.Dropout(0.0)  # Drop rate 0

        # Max Pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout3 = nn.Dropout(0.0)  # Drop rate 0

    def forward(self, x):
        # Model1
        x = self.model1(x)

        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x


# Create a torch model instance
torch_model = CustomCNN()
print(torch_model)

# Load weights from keras model
def extract_weights(model, weights_dict, prefix=''):
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # Check if the layer is a nested model
            extract_weights(layer, weights_dict, prefix=prefix + layer.name + '_')
        else:
            weights = layer.get_weights()
            if weights:
                # Print layer name and weights shapes
                print(f"Extracting weights for layer: {prefix + layer.name}")
                for i, w in enumerate(weights):
                    print(f"  Weight {i} shape: {w.shape}")


                # Verify expected shapes based on layer type
                if 'conv' in layer.name:
                    expected_shape = (layer.filters, layer.input_shape[-1], layer.kernel_size[0], layer.kernel_size[1])
                    if weights[0].shape != (expected_shape[2], expected_shape[3], expected_shape[1], expected_shape[0]):
                        print(
                            f"Warning: Unexpected shape for conv weights in {layer.name}. Expected: {expected_shape}, Got: {weights[0].shape}")
                elif 'dense' in layer.name:
                    expected_shape = (layer.input_shape[-1], layer.units)
                    if weights[0].shape != expected_shape:
                        print(
                            f"Warning: Unexpected shape for dense weights in {layer.name}. Expected: {expected_shape}, Got: {weights[0].shape}")
                weights_dict[prefix + layer.name] = weights

                # print(prefix + layer.name)

keras_model = load_model("inbreast_vgg16_512x1.h5")
keras_weights = {}
extract_weights(keras_model, keras_weights)

# Mapping of Keras layer names to PyTorch layers
name_mapping = {
    'model_1_block1_conv1': ('model1', 'conv1_1'),
    'model_1_block1_conv2': ('model1', 'conv1_2'),
    'model_1_block2_conv1': ('model1', 'conv2_1'),
    'model_1_block2_conv2': ('model1', 'conv2_2'),
    'model_1_block3_conv1': ('model1', 'conv3_1'),
    'model_1_block3_conv2': ('model1', 'conv3_2'),
    'model_1_block3_conv3': ('model1', 'conv3_3'),
    'model_1_block4_conv1': ('model1', 'conv4_1'),
    'model_1_block4_conv2': ('model1', 'conv4_2'),
    'model_1_block4_conv3': ('model1', 'conv4_3'),
    'model_1_block5_conv1': ('model1', 'conv5_1'),
    'model_1_block5_conv2': ('model1', 'conv5_2'),
    'model_1_block5_conv3': ('model1', 'conv5_3'),
    'conv2d_1': 'conv1',
    'batch_normalization_1': 'bn1',
    'conv2d_2': 'conv2',
    'batch_normalization_2': 'bn2'
}

# Function to convert Keras weights to PyTorch format
def convert_weights(keras_weights):
    if keras_weights.ndim == 4:  # Convolutional layer
        # Transpose the weights from (H, W, C_in, C_out) to (C_out, C_in, H, W)
        return torch.from_numpy(np.transpose(keras_weights, (3, 2, 0, 1)))
    elif keras_weights.ndim == 2:  # Dense layer
        # Transpose the weights from (input_features, output_features) to (output_features, input_features)
        return torch.from_numpy(np.transpose(keras_weights))
    else:
        return torch.from_numpy(keras_weights)

# # Add weights to torch model
for keras_name, torch_names in name_mapping.items():
    if isinstance(torch_names, tuple):  # submodules in layer model1
        module, layer = torch_names
        target_layer = getattr(getattr(torch_model, module), layer)
    else:
        target_layer = getattr(torch_model, torch_names)

    layer_weights = keras_weights[keras_name]
    conv_weights, conv_bias = layer_weights[0], layer_weights[1]

    # Depending on the layer type, different actions might be needed
    if 'conv' in torch_names[-1] or 'conv' in torch_names or 'fc' in torch_names:
        target_layer.weight.data = convert_weights(conv_weights)
        target_layer.bias.data = torch.from_numpy(conv_bias)
    elif 'bn' in torch_names:
        target_layer.weight.data = torch.from_numpy(layer_weights[0])
        target_layer.bias.data = torch.from_numpy(layer_weights[1])
        target_layer.running_mean = torch.from_numpy(layer_weights[2])
        target_layer.running_var = torch.from_numpy(layer_weights[3])

torch.save(torch_model, 'inbreast_vgg16_512x1.pth')