from utils import *
from model import *
import torch
import torch.nn as nn
import torch.optim as opt
import pickle
import torchvision as tv



def calc_acc(pred, act):
    with torch.no_grad():
        max_vals, max_indices = torch.max(pred, 1)
        train_acc = (max_indices == act.to(dtype=torch.long)).sum().item() / max_indices.size()[0]
    return train_acc


def train(model, tr_loader, epochs, ustep,
          va_loader, optimizer,
          loss_fn, metrics=calc_acc,
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          model_cp='./waights', train_profile=dict(), startWith=0):
    model.train()

    for ep in range(epochs):
        ep += startWith
        samples = 0
        epoch_loss = 0
        stp = 0
        epoch_acc = 0
        for img, lbl in tr_loader:

            img = img.to(device=device)
            lbl = lbl.to(device=device)
            samples += img.shape[0]
            pred = model(img)

            lss = loss_fn(pred, lbl)

            lss.backward()
            if samples > ustep:
                print(samples, 'netup')
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
                samples = 0

            epoch_loss += lss.item()
            stp += 1
            sacc = metrics(pred, lbl)
            epoch_acc += sacc
            print(stp, lss.item(), sacc)
        train_profile[ep] = [epoch_loss/ stp, epoch_acc / stp]

        # model.eval()
        vepoch_loss = 0
        vstp = 0
        vepoch_acc = 0
        with torch.no_grad():
            for img, lbl in va_loader:
                img = img.to(device=device)
                lbl = lbl.to(device=device)
                pred = model(img)

                vlss = loss_fn(pred, lbl)
                vepoch_loss += vlss.item()
                vstp += 1
                vepoch_acc += metrics(pred, lbl)
        train_profile[ep].append(vepoch_loss/ vstp)
        train_profile[ep].append(vepoch_acc / vstp)
        print(train_profile[ep])
        torch.save(model.state_dict(),
                   model_cp + '/weights' + str(ep) + '.pth')
        save_dhist(train_profile, model_cp + '/histInfo.pickle')
    return train_profile


def plting(dic):
    trloss = list()
    tracc = list()
    valoss = list()
    vaacc = list()
    for ep, va in dic.items():
        trl, tra, val, vaa = va
        trloss.append(trl)
        tracc.append(tra)
        valoss.append(val)
        vaacc.append(vaa)
    plt.subplot(2, 1, 1)
    plt.plot(trloss, label='tr_loss')
    plt.plot(valoss, label='va_loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(tracc, label='tr_acc')
    plt.plot(vaacc, label='va_acc')
    plt.legend()
    plt.show()


def save_dhist(di, name):
    with open(name, 'wb') as handle:
        pickle.dump(di, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dhist(name):
    with open(name, 'rb') as handle:
        b = pickle.load(handle)
    return b


if __name__ == '__main__'  :
    torch.cuda.empty_cache()
    layrDepth = [4*16, 4*32, 4*64, 4*64, 4*64, 4*16]
    N= len(layrDepth)
    model = eval_clfying((200, 200),
                         depth=layrDepth,
                         kernel_sizes=[None] *N , drop=[0] * N,
                         booling=[1] * N,
                         DFeature=lightConvDenseModel)
    model.cuda()

    hist = dict();
    stw = 0
    # 0ld model getinfog
    # stw 0 ;lr = 1e-3 ;bs = 128
    # stw 7 ;lr = 1e-3 ;bs = 256
    # stw 13;lr = 1e-4 ;bs = 256
    #hist = load_dhist('waights/histInfo.pickle'); stw = 13
    #model.load_state_dict(torch.load('waights/weights%i.pth'%(stw-1)))
    # Ent
    print(count_params(model))
    optim = opt.Adam(model.parameters(), 1e-4)

    tr_loader, va_loader = getloaders(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()

    tp = train(model, tr_loader,
               epochs=30, va_loader=va_loader,
               optimizer=optim, loss_fn=loss_fn,
               ustep=255, train_profile=hist, startWith=stw)

    save_dhist(tp, 'model5layer30ep.pickle')
    plting(tp)


'''
if __name__ == '__main__' :
    torch.cuda.empty_cache()
    model = tv.models.resnet34()
    model.fc = nn.Linear(512,2)

    model = torch.nn.Sequential(
        nn.BatchNorm2d(3), model
    )
    model.cuda()

    hist = dict();
    stw = 0
    hist = load_dhist('waights/histInfo.pickle'); stw = 13
    model.load_state_dict(torch.load('waights/weights%i.pth'%(stw-1)))
    print(count_params(model))
    optim = opt.Adam(model.parameters(), 1e-4)

    tr_loader, va_loader = getloaders(batch_size=64)

    loss_fn = nn.CrossEntropyLoss()

    tp = train(model, tr_loader,
               epochs=30, va_loader=va_loader,
               optimizer=optim, loss_fn=loss_fn,
               ustep=128, train_profile=hist, startWith=stw)

    save_dhist(tp, 'model5layer30ep.pickle')
    plting(tp)


'''


'''
if __name__ == '__main__' :
    torch.cuda.empty_cache()
    model = tv.models.resnet34()
    model.fc = nn.Linear(512,2)

    model = torch.nn.Sequential(
        nn.BatchNorm2d(3), model
    )
    model.cuda()

    hist = dict();
    stw = 0
    #hist = load_dhist('waights/histInfo.pickle'); stw = 15
    #model.load_state_dict(torch.load('waights/weights%i.pth'%(stw-1)))
    print(count_params(model))
    optim = opt.Adam(model.parameters(), 1e-4)

    tr_loader, va_loader = getloaders(batch_size=100)

    loss_fn = nn.CrossEntropyLoss()

    tp = train(model, tr_loader,
               epochs=30, va_loader=va_loader,
               optimizer=optim, loss_fn=loss_fn,
               ustep=125, train_profile=hist, startWith=stw)

    save_dhist(tp, 'model5layer30ep.pickle')
    plting(tp)

'''
'''if __name__ == '__main__' :
    torch.cuda.empty_cache()
    model = tv.models.resnet50()
    model.fc = nn.Linear(2048, 2)

    model = torch.nn.Sequential(
        nn.BatchNorm2d(3), model
    )
    model.cuda()

    hist = dict();
    stw = 0
    #hist = load_dhist('waights/histInfo.pickle'); stw = 15
    #model.load_state_dict(torch.load('waights/weights%i.pth'%(stw-1)))
    print(count_params(model))
    optim = opt.Adam(model.parameters(), 1e-4)

    tr_loader, va_loader = getloaders(batch_size=25)

    loss_fn = nn.CrossEntropyLoss()

    tp = train(model, tr_loader,
               epochs=30, va_loader=va_loader,
               optimizer=optim, loss_fn=loss_fn,
               ustep=125, train_profile=hist, startWith=stw)

    save_dhist(tp, 'model5layer30ep.pickle')
    plting(tp)
'''








