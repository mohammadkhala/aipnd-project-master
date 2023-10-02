import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_directory', type=str, help='path to the data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, choices=['vgg16', 'resnet18'], default='vgg16', help='choose architecture (vgg16 or resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    args = parser.parse_args()

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(args.data_directory, transform=data_transforms)

    # Split the dataset into training, validation, and testing sets
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # Load the pre-trained model
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[0].in_features
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features

    # Freeze pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    classifier = nn.Sequential(
        nn.Linear(num_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, len(dataset.classes)),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                valid_accuracy += torch.sum(preds == labels.data)

        # Calculate average losses and accuracy
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = valid_accuracy.double() / len(valid_loader.dataset)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Accuracy: {valid_accuracy:.4f}")

    # Save the model checkpoint
    checkpoint = {
        'arch': args.arch,
        'model': model,
        'state_dict': model.state_dict(),
        'class_to_idx': dataset.class_to_idx
    }
    torch.save(checkpoint, f"{args.save_dir}/checkpoint.pth")

if __name__ == '__main__':
    main()
