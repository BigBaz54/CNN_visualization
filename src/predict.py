import argparse
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from models.mnist_cnn import MnistCNN


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with a model')
    parser.add_argument('--model', type=str, default='MnistCNN', help='model to predict with', choices=['MnistCNN'])
    parser.add_argument('--model-file', type=str, help='path to model file', required=True)
    parser.add_argument('--size', type=int, default=3, help='number of samples to predict')

    return parser.parse_args()

def predict(model, dataset, size):
    model.eval()

    test_data = torch.utils.data.Subset(dataset, range(50000, 60000))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=size, shuffle=True)
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions = zip(images, predicted)

    return predictions


if __name__ == '__main__':
    args = parse_args()
    model = args.model
    model_file = args.model_file
    size = args.size

    transform = transforms.Compose([transforms.ToTensor()])

    if model == 'MnistCNN':
        dataset = datasets.MNIST(root='../data', download=True, transform=transform)
        model = MnistCNN()
    
    model.load_state_dict(torch.load(model_file))

    predictions = predict(model, dataset, size)

    _, axes = plt.subplots(1, size, figsize=(size * 3, 3))
    for i, (image, prediction) in enumerate(predictions):
        axes[i].imshow(image[0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Prediction: {prediction}')
    if not os.path.exists(os.path.join('..', 'temp')):
        os.makedirs(os.path.join('..', 'temp'))
    plt.savefig(os.path.join('..', 'temp', 'predictions.png'))
