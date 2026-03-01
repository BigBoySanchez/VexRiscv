import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import os
import argparse

def get_imagenet_categories():
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT
    return weights.meta["categories"]

def letterbox_image(image: Image.Image, expected_size=(224, 224)):
    iw, ih = image.size
    ew, eh = expected_size
    
    scale = min(ew / iw, eh / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    resample = getattr(Image, 'Resampling', Image)
    image = image.resize((nw, nh), resample.BILINEAR)
    
    new_image = Image.new('RGB', expected_size, (0, 0, 0))
    new_image.paste(image, ((ew - nw) // 2, (eh - nh) // 2))
    return new_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    args = parser.parse_args()
    img_path = args.img_path
    
    print(f"Loading image from {img_path}...")
    img = Image.open(img_path).convert('RGB')
    print(f"Original size: {img.size} (W, H)")
    
    img_letterboxed = letterbox_image(img, (224, 224))
    print(f"Letterboxed size: {img_letterboxed.size} (W, H)")
    
    out_path = os.path.splitext(img_path)[0] + '_letterboxed.jpg'
    img_letterboxed.save(out_path)
    print(f"Saved letterboxed image to {out_path}")
    
    tensor = F.to_tensor(img_letterboxed)
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0)
    
    print("Loading ResNet-50 model...")
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.eval()
    
    print("Running inference...")
    with torch.no_grad():
        out = model(tensor)
        
    logits = out[0]
    
    probs = torch.nn.functional.softmax(logits, dim=0)
    
    top_prob, top_classid = torch.max(probs, dim=0)
    top_classid = top_classid.item()
    top_prob = top_prob.item()
    
    categories = get_imagenet_categories()
    class_name = categories[top_classid]
    
    # print("\nFinal Logits:")
    # torch.set_printoptions(profile="full")
    # print(logits)
    # torch.set_printoptions(profile="default")
    
    print(f"\nFinal Prediction:")
    print(f"ID: {top_classid}")
    print(f"String: {class_name}")
    print(f"Confidence: {top_prob:.4f}")

if __name__ == '__main__':
    main()
