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
