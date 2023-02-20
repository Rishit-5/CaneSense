# import torch
# import torchvision
# from matplotlib import pyplot as plt
# from torch.testing._internal.common_quantization import accuracy
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
#
#
# #train and test data directory
# data_dir = "images/train"
# test_data_dir = "images/test"
#
#
# #load the train and test data
# dataset = ImageFolder(data_dir,transform = transforms.Compose([
#     transforms.Resize((64,64)),transforms.ToTensor()
# ]))
# test_dataset = ImageFolder(test_data_dir,transforms.Compose([
#     transforms.Resize((64,64)),transforms.ToTensor()
# ]))
#
# img, label = dataset[0]
# print(img.shape,label)
#
# #output :
# #torch.Size([3, 150, 150]) 0
#
# print("Follwing classes are there : \n",dataset.classes)
#
# def display_img(img,label):
#     print(f"Label : {dataset.classes[label]}")
#     plt.imshow(img.permute(1,2,0))
#     plt.show()
#
# #display the first image in the dataset
# # display_img(*dataset[0])
#
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import random_split
#
# batch_size = 128
# val_size = int(len(dataset)*.2)
# train_size = len(dataset) - val_size
#
# train_data,val_data = random_split(dataset,[train_size,val_size])
# print(f"Length of Train Data : {len(train_data)}")
# print(f"Length of Validation Data : {len(val_data)}")
#
# #output
# #Length of Train Data : 12034
# #Length of Validation Data : 2000
#
# #load the train and validation into batches.
# train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
# val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ImageClassificationBase(nn.Module):
#
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)  # Generate predictions
#         loss = F.cross_entropy(out, labels)  # Calculate loss
#         return loss
#
#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)  # Generate predictions
#         loss = F.cross_entropy(out, labels)  # Calculate loss
#         acc = accuracy(out, labels)  # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
#
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
#
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['val_loss'], result['val_acc']))
#
#
# class NaturalSceneClassification(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Flatten(),
#             nn.Linear(82944, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 6)
#         )
#
#     def forward(self, xb):
#         return self.network(xb)
#
#
# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#
#
# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)
#
#
# def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
#     history = []
#     optimizer = opt_func(model.parameters(), lr)
#     for epoch in range(epochs):
#
#         model.train()
#         train_losses = []
#         for batch in train_loader:
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         model.epoch_end(epoch, result)
#         history.append(result)
#
#     return history
#
# num_epochs = 30
# opt_func = torch.optim.Adam
# model = NaturalSceneClassification()
# lr = 0.001
# #fitting the model on training data and record the result after each epoch
# history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms


class CustomImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length


class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, 6)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
def main():

    hyper_param_epoch = 2
    hyper_param_batch = 8
    hyper_param_learning_rate = 0.001

    transforms_train = transforms.Compose([transforms.Resize((64, 64)),
                                           transforms.RandomRotation(10.),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor()])

    train_data_set = CustomImageDataset(data_set_path="AugmentedImages/train", transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path="AugmentedImages/test", transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

    if not (train_data_set.num_classes == test_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = train_data_set.num_classes
    net = CustomConvNet(num_classes)
    # custom_model = net(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for e in range(hyper_param_epoch):
        for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % hyper_param_batch == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(e + 1, hyper_param_epoch, loss.item()))

    # Test the model
    # Specify a path
    PATH = "state_dict_model.pt"

    # Save
    torch.save(net.state_dict(), PATH)
# Load
#     model = CustomConvNet(6)
#     model.load_state_dict(torch.load(PATH))
#     model.eval()
#     custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for item in test_loader:
#             images = item['image'].to(device)
#             labels = item['label'].to(device)
#             outputs = custom_model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += len(labels)
#             correct += (predicted == labels).sum().item()
#
#         print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
# #
# #put in folder with images from google and then train dataset

if __name__ == '__main__':
    main()