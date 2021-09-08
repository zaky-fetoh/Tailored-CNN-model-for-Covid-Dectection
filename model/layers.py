import torch.nn.functional as F
import torch.nn as nn
import torch


def get_padding(kernel_size):
    padding = [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
               for k in kernel_size[::-1]]
    pads = [padding[i][j] for i in range(2) for j in range(2)]
    return pads


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SameConv2d(nn.Module):
    def __init__(self, in_chunnels, out_chunnels, kernel_size, group=1):
        super().__init__()
        self.in_chunnels, self.out_chunnels, self.kernel_size, self.group = in_chunnels, out_chunnels, kernel_size, group
        self.conv = nn.Conv2d(in_chunnels, out_chunnels, kernel_size, groups=group)
        self.pads = get_padding(self.kernel_size)

    def forward(self, X):
        X = F.pad(X, self.pads)
        return self.conv(X)


class Conv2DLinSepKer(nn.Module):
    def __init__(self, in_chunnels, out_chunnels, kernel_len):
        super().__init__()
        self.in_chunnels, self.out_chunnels, self.kernel_len = in_chunnels, out_chunnels, kernel_len

        self.X_conv = SameConv2d(in_chunnels, in_chunnels, (kernel_len, 1), in_chunnels)
        self.Y_conv = SameConv2d(in_chunnels, in_chunnels, (1, kernel_len), in_chunnels)
        self.pointwise = nn.Conv2d(in_chunnels, out_chunnels, 1)

    def forward(self, X):
        return self.pointwise(self.Y_conv(self.X_conv(X)))


class lightConvDenseModel(nn.Module):
    def __init__(self, in_chunnels, out_chunnels,
                 kernel_lens=[3, 5, 7, 9], intract=None):
        super().__init__()

        self.lis = nn.ModuleList()
        single_out = out_chunnels // len(kernel_lens)

        if intract is None:
            self.bn = nn.BatchNorm2d(single_out)
            self.intract = lambda x: F.leaky_relu(self.bn(x))

        self.lis.append(Conv2DLinSepKer(in_chunnels, single_out, kernel_lens[0]))
        for ln in kernel_lens[1:]:
            self.lis.append(Conv2DLinSepKer(single_out, single_out, ln))

    def forward(self, X):
        cats = self.intract(self.lis[0](X))
        X = cats
        for lyr in self.lis[1:]:
            o = self.intract(lyr(X))
            cats = torch.cat([cats, o], dim=1)
            X = o
        return cats


class lightFB(nn.Module):
    def __init__(self, in_chunnel, out_chunnel, kernel_size = None,
                 booling=True, drop=.5,
                 activation_func=F.leaky_relu,
                 DFeature=lightConvDenseModel):
        super().__init__()
        self.activation_func, self.booling, self.drop = activation_func, booling, drop

        if self.drop:
            self.dr = nn.Dropout2d(self.drop)

        if kernel_size is None:
            self.feature = DFeature(in_chunnel, out_chunnel)
        else:
            self.feature = DFeature(in_chunnel, out_chunnel,
                                    kernel_size)

        self.bn = nn.BatchNorm2d(out_chunnel)

    def forward(self, X):
        X = self.feature(X)
        if self.drop:
            X = self.dr(X)
        X = self.bn(X)
        if self.booling:
            X = F.max_pool2d(X, 2)
        X = self.activation_func(X)
        return X


count_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    lyr = Conv2DLinSepKer(1, 50, 5)
    dm = lightConvDenseModel(1, 64 * 4)
    inp = torch.rand(3, 1, 100, 100)
    out = dm(inp)
    print(count_params(dm))
    print(out)
    '''
    entlayer = EntopyLayer(1, 5, kernel_size=5)
    aventlayer = EntopyLayer(1, 5, False, 5)
    hog1 = AhogLayer(1, 6, (3, 3), True, (5, 5))
    hog2 = AhogLayer(1, 6, (5, 5), False, (5, 5))
    harr = Aharris(1, 8, (5, 5))

    d = DeepFeature(1, 5, (3, 3))
    fx = FeatureBlock(1, 5, (3, 3))
    inp = torch.rand(3, 1, 100, 100)
    o1 = fx(inp)'''
    coun = sum(p.numel() for p in fx.parameters())
