# ResNet-9
ResNet-9 is a deep convolutional neural network trained on the CIFAR-10 dataset. The architecture is implemented from the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf), it's a residual learning network to ease the training of networks that are substantially deeper. I designed a smalled architecture compared to the paper and achieved 93.65% testing accuracy on the CIFAR-10 dataset with significantly less training time. Some more modifications have also been made which are different from the paper those are maxpooling has been used for the downsampling so only the important features are selected, ReLU activations has been used on every Convolution layer instead of activation on every other layer, and a dropout of 20% has been used. These are used with Adam optimizer using OneCycleLR as the learning rate scheduler instead of SGD with COSINE Annealing  as mentioned in the paper.
# Dependencies
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [PIL](https://pypi.org/project/Pillow/)
* [Numpy](https://numpy.org/)
* [OS](https://docs.python.org/3/library/os.html)
* [fast.ai](https://www.fast.ai/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [torchinfo](https://github.com/TylerYep/torchinfo)

Once you have these dependencies installed, you can clone the Custom ResNet-9 repository from GitHub:
```bash
https://github.com/Moddy2024/ResNet-9.git
```
# Key Files
* [resnet-9.ipynb](https://github.com/Moddy2024/ResNet-9/blob/main/resnet-9.ipynb) - This file shows how the dataset has been downloaded, how the data looks like, the transformations, data augmentations, architecture of the ResNet and the training.
* [prediction.ipynb](https://github.com/Moddy2024/ResNet-9/blob/main/prediction.ipynb) - This file loads the trained model file and shows how to do predictions on single images, multiple images contained in a folder and images(multiple or single) that can be uploaded to google colab temporarily to perform the prediction.
* [trained-models](https://github.com/Moddy2024/ResNet-9/tree/main/trained-models) - This directory contains the best trained model and the trained model saved after the last epoch.
* [test-images](https://github.com/Moddy2024/ResNet-9/tree/main/test-images) - This directory contains test images collected randomly from the internet of different categories, sizes and shape for performing the predictions and seeing the results.
# Features
This custom ResNet-9 includes the following features:

* Support for multiple image sizes and aspect ratios
* Option to fine-tune the model on a specific dataset
* Ability to save and load trained models
# Training and Validation Image Statistics
The dataset used to train the model is CIFAR-10. The CIFAR-10 dataset consists of 60,000 32x32 color training images and 10,000 test images. Each image is labeled with one of 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. There are 6,000 images of each class in the training set, and 1,000 images of each class in the test set. CIFAR-10 is a popular choice for benchmarking because it is a well-defined and widely-used dataset, and the images are small enough that it is possible to train relatively large models on a single machine.
# Dataset
The  CIFAR-10 dataset (Canadian Institute For Advanced Research) can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). It can also be downloaded from PyTorch Datasets.
```bash
# Load the CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
```
In this repository the dataset has been downloaded using fast.ai as seen below.
```bash
from fastai.data.external import untar_data, URLs
data_dir = untar_data(URLs.CIFAR)
data_dir = str(data_dir)
```
