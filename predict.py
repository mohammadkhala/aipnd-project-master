import argparse
from email.policy import strict

import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    return model, class_to_idx

def process_image(image):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return image_transforms(image)

def predict(image_path, model, class_to_idx, topk=5):
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)

        top_probabilities, top_indices = probabilities.topk(topk)
        top_probabilities = top_probabilities[0].tolist()
        top_indices = top_indices[0].tolist()

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        top_classes = [idx_to_class[index] for index in top_indices]
        top_flower_names = [cat_to_name[class_name] for class_name in top_classes]

    return top_probabilities, top_flower_names

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict the flower name from an image')
    parser.add_argument('image_path', type=str, help='path to the input image')
    parser.add_argument('checkpoint', type=str, help='path to the trained model checkpoint')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to the category names mapping')
    parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference if available')
    args = parser.parse_args()

    # Load the category names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f,strict=False)

    # Load the trained model
    model, class_to_idx = load_checkpoint(args.checkpoint)

    # Set the device to GPU if available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Predict the class probabilities and names
    top_probabilities, top_flower_names = predict(args.image_path, model, class_to_idx, args.top_k)

    # Print the results
    for i in range(args.top_k):
        print(f"Flower Name: {top_flower_names[i]} | Probability: {top_probabilities[i]:.3f}")
