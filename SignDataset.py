# Test the model
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from main import CustomConvNet, CustomImageDataset

PATH = "state_dict_model.pt"
transforms_test = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ToTensor()])
hyper_param_batch = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_data_set = CustomImageDataset(data_set_path="AugmentedImages/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

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

    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    print("Total incorrect: " + str(total-correct))

#put in folder with images from google and then train dataset
# import torch
# import torchvision
# import torchvision.transforms as transforms
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
#
#
# from PIL import Image
#
# from main import CustomImageDataset
#
# from main import CustomConvNet as Net
#
#
#
#
# def SignDataset():
#     transforms_test = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
#
#     testset = CustomImageDataset(data_set_path="AugmentedImages/test", transforms=transforms_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                              shuffle=True, num_workers=2)
#
#
#     #load neural net
#     PATH = 'state_dict_model.pt'
#     net = Net(6)
#     net.load_state_dict(torch.load(PATH))
#
#     X = testset.transformed_images()
#     print(X[0].shape)
#
#
#
#
#     #total accuracy
#
#     correct = 0
#     total = 0
#     # since we're not training, we don't need to calculate the gradients for our outputs
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             # calculate outputs by running images through the network
#             outputs = net(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
#
#     #accuracy per class
#
#     # prepare to count predictions for each class
#
#     # again no gradients needed
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = net(images)
#             _, predictions = torch.max(outputs, 1)
#             # collect the correct predictions for each class
#
#
# if __name__ == '__main__':
#     SignDataset()