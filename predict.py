import os
import torch
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def combine_patches(image, patches):
    w, h = image[0].shape
    seg = torch.zeros(w, h)
    x, y = (0, 0) 
    for patch in patches:
        if y + SIZE > h:
            y = 0
            x += SIZE
        seg[x:x + SIZE, y:y + SIZE] = patch
        y += SIZE
    return seg

def post_process_image(pred):
    classes = np.unique(pred)
    no_of_colors = len(classes)
    colors = [np.random.choice(range(255),size=3) for _ in range(no_of_colors)]
    r, c = pred.shape
    img = np.empty((r, c, 3), dtype=np.uint8)
    for i, cls in enumerate(classes):
        img[pred == cls] = colors[i]
    
    return img


def main(image_path, to_path="."):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Resize((SIZE,SIZE)),
        transforms.ToTensor()
    ])

    model = torch.load("./checkpoint/model", map_location=device)
    model.eval()

    image = transform(Image.open(image_path).convert('RGB'))

    width_t = (image[0].shape[0] // SIZE) * SIZE
    height_t = (image[0].shape[1] // SIZE) * SIZE

    image = image[:, 0:width_t, 0:height_t]

    patches = image.unfold(0, 3, 3)
    patches = patches.unfold(1, SIZE, SIZE)
    patches = patches.unfold(2, SIZE, SIZE)
    batch = patches.reshape(-1, 3, SIZE, SIZE)

    batch = batch.to(device)

    batch_seg = model.forward_encoder(batch)
    batch_seg = torch.argmax(batch_seg, axis=1).float()

    pred_seg = combine_patches(image, batch_seg)
    pred = pred_seg.int().cpu().numpy()

    image_name = image_path.split(os.path.sep)[-1].split(".")[0]
    Image.fromarray(post_process_image(pred)).save(f"{to_path}{os.path.sep}{image_name}-seg.png")

if __name__ == "__main__":
    SIZE = 224
    import sys
    try:
        image_path = sys.argv[1]
    except IndexError:
        print("Usage:\n\t python predict.py [<image_path>/<dir_path>]")
    if os.path.isdir(image_path):
        os.makedirs("./output", exist_ok=True)
        for img_path in os.listdir(image_path):
            img = os.path.join(image_path, img_path)
            main(img, "./output")
    else:
        main(image_path)
