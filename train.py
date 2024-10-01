import torch
import torch.nn as nn
from torchvision import transforms
from model import WNet
from n_cut_loss import soft_n_cut_loss
from dataset import WNetData



if __name__ == "__main__":
    SIZE = 224
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 250

    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.Resize(128),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(128),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor()
    ])

    train_dataset = WNetData("./dataset/BSDS500", "train", train_transform)
    val_dataset   = WNetData("./dataset/BSDS500", "val", val_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4, num_workers=2, shuffle=False)

    model = WNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    loss = nn.BCELoss()
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, [inputs, outputs] in enumerate(train_dataloader, 0):

            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()

            seg, recons = model(inputs)

            l_soft_n_cut = soft_n_cut_loss(inputs, seg)
            l_reconstruction = loss(inputs,recons)

            loss = (l_reconstruction + l_soft_n_cut)
            loss.backward(retain_graph=False)
            optimizer.step()

            if (i%50) == 0:
                print(i)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")

        torch.save(model, "./checkpoint/model")