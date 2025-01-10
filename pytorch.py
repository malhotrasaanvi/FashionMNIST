import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    set = ""
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if (training == True):
        set=datasets.FashionMNIST('./data',train=training, download=True,transform=custom_transform)
    else:
        set=datasets.FashionMNIST('./data', train=training, transform=custom_transform)
    loader = torch.utils.data.DataLoader(set, batch_size=64, shuffle=True)
    return loader

def build_model():
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
    )
    return model

def train_model(model, train_loader, criterion, T=5):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    for epoch in range(T):
        correct = 0
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%) Loss: {avg_loss:.3f}")

def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

   
    


def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    image = test_images[index]
    logits = model(image)
    prob = F.softmax(logits, 1)
    top_p, top_class = prob.topk(3, 1)
    top_p = top_p.tolist()[0]
    top_class = top_class.tolist()[0]
    for x in range(len(top_p)):
        print(f'{class_names[top_class[x]]}: {top_p[x] * 100:.2f}%')

   


if __name__ == '__main__':
   
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = False)
    N=100
    test_images = torch.randn(N, 1, 28, 28)
    predict_label(model, test_images, 1)
