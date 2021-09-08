import torch.utils.data as udata
import torchvision.transforms as trans
import os.path as path
from glob import glob
import PIL.Image as Image
import torch
import PIL
import os


class covid(udata.Dataset):

    def __init__(self, dataset_path,
                 transform=trans.Compose([trans.ToTensor()
                                          ]),
                 reshape=(200, 200)):
        self.class_labels = os.listdir(dataset_path)
        self.class_paths = {x: path.join(dataset_path, x)
                            for x in self.class_labels}
        self.transform = transform
        self.reshape = reshape

        self.ids = {x: list() for x in self.class_labels}
        for clss, pth in self.class_paths.items():
            for item in glob(pth + '\*.png'):
                self.ids[clss].append((
                    path.join(pth, item),
                    self.class_labels.index(clss)
                ))

        min_num = len(self.ids[list(self.ids.keys())[0]])
        for clss, lis in self.ids.items():
            if len(lis) < min_num:
                min_num = len(lis)
        lis = list()
        for _, li in self.ids.items():
            lis += li[:min_num]
        self.items = lis

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        pth, lbl = self.items[item]
        img = Image.open(pth)
        img = img.resize(self.reshape)
        #img = img.convert('RGB')
        img = self.transform(img)
        return (img, lbl)


def getloaders(batch_size=64, valradio=.3,
               shuffle=True, num_workers=2, pin_memory=True,
               dts=covid('E:\covdt_CoCo\initilW0Rk\ClassificationUsingCNN\data')):
    n_valid = int(len(dts) * valradio)
    n_train = len(dts) - n_valid
    train, valid = udata.random_split(dts, [n_train, n_valid])
    tloader = udata.DataLoader(train, batch_size, shuffle,
                               num_workers=num_workers,
                               pin_memory=pin_memory)
    vloader = udata.DataLoader(valid, batch_size, shuffle,
                               num_workers=num_workers,
                               pin_memory=pin_memory)
    return tloader, vloader


if __name__ == '__main__':
    tl, vl = getloaders()
    vv = iter(tl).next()
