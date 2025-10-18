'''
AD/NC ConvNeXt Prediction Script

This script loads a trained ConvNeXt model and predicts labels for AD/NC images.
It supports two modes:
- auto: randomly select images from the dataset test set
- custom: input image paths manually

Features:
- Supports GPU/CPU prediction
- Visualizes images with predicted labels and confidence
- Prints summary including number of correct predictions if labels are available
'''

import os
import torch
import argparse
import random
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dataset import DataLoader as CustomDataLoader
from utils import convnext_small

# ---------------------------- Prediction Function ----------------------------
def predict(model, device, images, class_names=['CN','AD'], visualize=True):
    """
    Predict labels for a list of images using a trained model.

    Args:
        model: trained ConvNeXt model
        device: 'cuda' or 'cpu'
        images: list of images or (image, label) tuples
        class_names: list of class names for display
        visualize: whether to show images with predictions

    Returns:
        results: list of dicts with keys 'pred', 'conf', 'label'
    """
    model.eval()  # set model to evaluation mode

    # -- Transform for input images --
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale images
    ])
    
    results = []

    # -- Iterate through images --
    with torch.no_grad():  # no gradient computation needed
        for item in images:
            if isinstance(item, tuple):
                img, label = item  # image + true label
            else:
                img, label = item, None  # image only

            # -- Convert to tensor if not already --
            if not isinstance(img, torch.Tensor):
                img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension
            else:
                img_tensor = img.unsqueeze(0).to(device)

            # -- Model prediction --
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)  # probabilities
            conf, pred = prob.max(dim=1)  # predicted class and confidence

            results.append({'pred': pred.item(), 'conf': conf.item(), 'label': label})

            # -- Visualization --
            if visualize:
                plt.imshow(img if isinstance(img, Image.Image) else img.squeeze().cpu(), cmap='gray')
                title = f"Pred: {class_names[pred.item()]} ({conf.item():.2f})"
                if label is not None:
                    title += f" | Label: {class_names[label]}"
                plt.title(title)
                plt.axis('off')
                plt.show()
    
    return results

# ---------------------------- Load Model Function ----------------------------
def load_model(model_path, device='cuda'):
    """
    Load a trained ConvNeXt model from checkpoint.
    """
    print('Loading model...')
    model = convnext_small(in_chans=1, num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ---------------------------- Main Script ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict images with trained ConvNeXt model")
    parser.add_argument('--model_path', type=str, default=r"recognition\ADNI_classification_49085949\results\small-in22k_best.pth")
    parser.add_argument('--datapath', type=str, default=r"D:\MachineLearningData\AD_NC")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_images', type=int, default=6, help='Number of images to predict')  # number of images to select
    parser.add_argument('--mode', type=str, default='auto', choices=['auto','custom'])  # auto or custom input
    args = parser.parse_args()

    class_names = ['CN', 'AD']
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Load model ----
    model = load_model(args.model_path, device=device)

    # ---- Prepare images ----
    images = []
    if args.mode == 'auto':
        # -- Load test dataset automatically --
        loader = CustomDataLoader(datapath=args.datapath, batch_size=1, num_workers=4, pin_memory=True)
        loader.load_data()
        _, _, test_loader = loader.get_loaders()
        
        all_test_samples = []
        for imgs, labels in test_loader:
            all_test_samples.extend([(imgs[i], labels[i].item()) for i in range(len(imgs))])
        
        sample_count = min(args.num_images, len(all_test_samples))  # select N images
        images = random.sample(all_test_samples, sample_count)

    else:  # custom mode
        # -- Input image paths manually --
        paths = input('Input image path(s) (comma separated if multiple): ')
        image_paths = [p.strip() for p in paths.split(',') if os.path.isfile(p.strip())]
        if not image_paths:
            raise ValueError("No valid image paths provided.")
        for path in image_paths[:args.num_images]:
            images.append(Image.open(path).convert('L'))

    # ---- Predict ----
    results = predict(model, device, images, class_names=class_names, visualize=True)
    
    # ---- Print summary ----
    correct_count = 0
    total_count = 0
    print("\n===== Prediction Results =====")
    for idx, res in enumerate(results):
        pred_name = class_names[res['pred']]
        label_name = class_names[res['label']] if res['label'] is not None else "N/A"
        print(f"Image {idx+1}: Pred: {pred_name} | Confidence: {res['conf']:.2f} | Label: {label_name}")
        # -- Count correct predictions if label is available --
        if res['label'] is not None:
            total_count += 1
            if res['pred'] == res['label']:
                correct_count += 1

    if total_count > 0:
        print(f"\nAccuracy: {correct_count}/{total_count} correct")  # summary accuracy

if __name__ == "__main__":
    main()
