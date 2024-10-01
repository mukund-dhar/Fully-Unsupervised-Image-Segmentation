import torch


SIZE = 224

def soft_n_cut_loss(inputs, segmentations):
    loss = 0
    for i in range(inputs.shape[0]):
        flatten_image = torch.mean(inputs[i], dim=0)
        flatten_image = flatten_image.reshape(flatten_image.shape[0]**2)
        loss += soft_n_cut_loss_(flatten_image, segmentations[i], 64, SIZE, SIZE)
    loss = loss / inputs.shape[0]
    return loss