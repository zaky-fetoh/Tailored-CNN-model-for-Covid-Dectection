import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from model import *
from trian import *
import PIL.Image as Image


def get_active(model, img, device='cuda'):
    #img = img.to(device=device)
    lis = list()
    for i in range(model.ConvBase.__len__() - 1):
        print(i)
        lis.append(model.ConvBase[:i+1](img))

    for i in range(lis.__len__()):
        lis[i] = vutils.make_grid(lis[i].permute(1, 0, 2, 3),
                                  padding=1).detach().cpu().numpy()

    return lis


normalize = lambda x: (x - x.min()) / (x.max() - x.min())


def save_figs(np_list, name, pth='visualize/'):
    for i, img in enumerate(np_list):
        img.shape = img.shape[1:] + (img.shape[0],)
        plt.imsave(pth + name + "_" + str(i) + ".png", img)


def prepro(img, single=True,d= 'cuda', transform=trans.Compose([trans.ToTensor()
                                                      ]), reshape=(200, 200)):
    img = img.resize(reshape)
    img = transform(img)
    if single:
        img = img.unsqueeze(0)
    return img.to(device= d)


def get_model(d = 'cuda', mnum= 21):
    layrDepth = [4 * 16, 4 * 32, 4 * 64, 4 * 64, 4 * 64, 4 * 16]
    N = len(layrDepth)
    model = eval_clfying((200, 200),
                         depth=layrDepth,
                         kernel_sizes=[None] * N, drop=[0] * N,
                         booling=[1] * N,
                         DFeature=lightConvDenseModel)
    model.to(device=d)
    model.load_state_dict(torch.load('waights/weights%i.pth' % (mnum),
                                     map_location=torch.device(d)))
    return model
if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = get_model()
    # Ent
    print(count_params(model))
    cov = Image.open(r'E:\covdt_CoCo\initilW0Rk\ClassificationUsingCNN\data\covid\COVID-13.png')
    nor = Image.open(r'E:\covdt_CoCo\initilW0Rk\ClassificationUsingCNN\data\norm\Normal-747.png')

    cov, nor = [prepro(im) for im in [cov, nor]]

    acti_cov = get_active(model, cov)
    acti_nor = get_active(model, nor)

    save_figs(acti_cov, 'cov')
    save_figs(acti_nor, 'nor')
