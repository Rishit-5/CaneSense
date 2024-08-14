from PIL import Image
import os, os.path

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from main import CustomConvNet as Net

import matplotlib.pyplot as plt
import numpy as np

classes = {0: 'stop', 1: 'yield', 2:'speed_limit', 3: 'rfrf', 4: 'sfcs', 5:'sdvdvs'}


def setup():
    # retrieve image data
    imgs = []
    path = 'wisc_sign_data'
    valid_images = [".jpeg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(Image.open(os.path.join(path, f)))

    # transform data
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor()
                                    ])
    image_tensors = [transform(img) for img in imgs]

    # load nn
    PATH = 'state_dict_model.pt'
    net = Net(6)
    net.load_state_dict(torch.load(PATH))

    return imgs, image_tensors, net


def predict(tensors, model):
    outputs = [model(tensor.unsqueeze(0)).clone().detach() for tensor in tensors]
    predictions = []
    for output in outputs:
        _, prediction = torch.max(output, 1)
        predictions.append(prediction)
    return predictions, outputs


def multi_plot(images, labels):
    fig = plt.figure(figsize=(20, 10))
    rows = 1
    columns = len(labels)
    for i in range(len(labels)):
        fig.add_subplot(rows, columns, i+1)
        npimg = images[i].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(classes[labels[i].item()])
    plt.show()


def main():
    images, image_tensors, net = setup()

    # make predictions
    predictions, outputs = predict(image_tensors, net)
    for pred, out in zip(predictions, outputs):
        print(out, pred)

    # show images
    multi_plot(image_tensors, predictions)


if __name__ == '__main__':
    main()
