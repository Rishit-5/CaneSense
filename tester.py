# Test the model
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from main import CustomConvNet, CustomImageDataset
import time as time

PATH = "state_dict_model.pt"
transforms_test = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ToTensor()])
hyper_param_batch = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_data_set = CustomImageDataset(data_set_path="surfaces/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

time1 = time.time()
model = CustomConvNet(test_data_set.num_classes).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for item in test_loader:
        images = item['image'].to(device)
        labels = item['label'].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        # print("predicted: " + str(predicted[0].item()))
        # print("label: " + str(labels[0].item()))
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    print("Total incorrect: " + str(total-correct))
time2 = time.time()

# Input to the model

print("Time: " + str(time2-time1))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess the input image
image_path = 'surfaces/train/water_concrete_severe/202204291633212-water-concrete-severe.jpg'
image = Image.open(image_path).convert('RGB')

# Define the transformations
preprocess = transforms.Compose([
    transforms.Resize((360, 240)),
    transforms.ToTensor(),
])

# Apply the transformations
input_image = preprocess(image).unsqueeze(0)  # Add batch dimension
input_image = input_image.requires_grad_()  # Enable gradient tracking

# Forward pass
output = model(input_image)

# Get the index of the predicted class
predicted_class = torch.argmax(output).item()

# Backward pass to compute gradients
model.zero_grad()
output[0, predicted_class].backward()

# Compute the saliency map
saliency_map = input_image.grad.data.abs().max(dim=1)[0].squeeze().numpy()

# Normalize the saliency map to the range [0, 1]
normalized_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

# Choose a colormap (e.g., 'jet', 'viridis', 'hot', etc.)
colormap = plt.get_cmap('jet')

# Apply the colormap to the normalized saliency map
colored_saliency_map = colormap(normalized_saliency_map)

# Convert the RGBA image to RGB
colored_saliency_map_rgb = colored_saliency_map[:, :, :3]

# Convert the RGB image to a PyTorch tensor
colored_saliency_map_tensor = transforms.ToTensor()(colored_saliency_map_rgb)

# Visualize the saliency map
plt.imshow(colored_saliency_map_tensor.permute(1, 2, 0).numpy())
plt.axis('off')
plt.show()
