from models.mnist_cnn import MnistCNN

import argparse
import torch
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, default='MnistCNN', help='model to train', choices=['MnistCNN'])
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    return parser.parse_args()  

def train(model, dataset, batch_size, epochs, criterion, optimizer):
    train_data = torch.utils.data.Subset(dataset, range(50000))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        val_acc = evaluate(model, dataset, 1000)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Val. acc.: {val_acc}')

    torch.save(model.state_dict(), model.__class__.__name__ + '.pth')

def evaluate(model, dataset, test_sample_size):
    model.eval()

    test_data = torch.utils.data.Subset(dataset, range(50000, 60000))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_sample_size, shuffle=True)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    model.train()

    return correct / total


if __name__ == '__main__':
    args = parse_args()
    model = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    transform = transforms.Compose([transforms.ToTensor()])

    if model == 'MnistCNN':
        dataset = datasets.MNIST(root='../data', download=True, transform=transform)
        model = MnistCNN()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(model, dataset, batch_size, epochs, criterion, optimizer)
