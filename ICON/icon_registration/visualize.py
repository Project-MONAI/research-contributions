import torch
import torchvision
import matplotlib.pyplot as plt
import footsteps

def show(tensor):
    plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach())
    plt.xticks([])
    plt.yticks([])

def render(im):
    if len(im.shape) == 5:
        im = im[:, :, :, :, im.shape[4] // 2]
    if torch.min(im) < 0:
        im = im - torch.min(im)
    if torch.max(im) > 1:
        im = im / torch.max(im)
    return im[:4, [0, 0, 0]].detach().cpu()


def plot_registration_result(image_A, image_B, net) :

    res = net(image_A, image_B)
    print(res.keys())



    
    plt.subplot(2, 2, 1)
    show(image_A)
    plt.subplot(2, 2, 2)
    show(image_B)
    plt.subplot(2, 2, 3)
    show(res["warped_image_A"])

    phi_AB_vectorfield = res["phi_AB"](net.identity_map)
    plt.contour(torchvision.utils.make_grid(phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach())
    plt.contour(torchvision.utils.make_grid(phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach())
    plt.subplot(2, 2, 4)
    show(res["warped_image_A"]- image_B)
    plt.tight_layout()
    footsteps.plot()
