import torch
import torch.nn
from model import *



def getkerneltuple(k):
    if isinstance(k, int):
        return (k, k)


def getflt_dim(inputshape, depth, booling):
    a, c = inputshape
    for b in booling:
        if b:
            a //= 2
            c //= 2

    return int(a * c * depth[-1])


class eval_clfying(nn.Module):
    def __init__(self, inshape, in_chunnel=1, num_clss=2,
                 depth=[32, 64, 128, 256, 32],
                 kernel_sizes=[3, 3, 3, 3, 3],
                 booling=[1, 1, 1, 1, 1],
                 drop=[0, 0, 0, 0, 0],
                 DFeature =lightConvDenseModel):
        super().__init__()
        lis = list()
        self.in_chunnel = in_chunnel
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.booling = booling
        self.drop = drop


        lis.append(nn.BatchNorm2d(in_chunnel))
        for i in range(len(depth)):
            lis.append(lightFB(in_chunnel, depth[i],
                                    getkerneltuple(kernel_sizes[i]),
                                    booling[i], drop[i],
                                    DFeature=DFeature))
            in_chunnel = depth[i]
        self.flttin = getflt_dim(inshape, depth, booling)
        self.ConvBase = nn.Sequential(*lis)
        self.DenseBase = nn.Sequential(
            nn.Linear(self.flttin, 64), nn.LeakyReLU(),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Linear(64, num_clss)
        )

    def forward(self, X):
        state = self.ConvBase(X)
        state = state.view(state.shape[0], -1)
        return self.DenseBase(state)


