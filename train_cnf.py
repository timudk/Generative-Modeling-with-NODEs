import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms

import cv2

import integrate

import lib.models as models
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument('--data',
                    choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', 'mnist'],
                    type=str, default='moons')
parser.add_argument('--dims', type=str, default='64,64,10')
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--num_bits', type=int, default=5)

parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--data_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--lr_max', type=float, default=1e-3)
parser.add_argument('--lr_min', type=float, default=1e-3)
parser.add_argument('--lr_interval', type=float, default=2000)
parser.add_argument('--weight_decay', type=float, default=1e-6)

parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def update_lr(optimizer, itr):
    lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1 + np.cos(itr / args.num_epochs * np.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_dataset(args):
    if args.data == "mnist":
        trans = tforms.Compose([tforms.ToTensor(), utils.Preprocess(args.num_bits), lambda x: x.view(28 ** 2)])
        train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
        test_set = dset.MNIST(root='./data', train=False, transform=trans, download=True)
        # im = train_set[0][0].numpy().reshape(28, 28, 1)
        # print(im, im.min(), im.max(), im.shape)
        # cv2.imshow("im", ((im + .5) * 255).astype(np.uint8))
        # cv2.waitKey(0)
        # 1/0
    else:
        dataset = toy_data.inf_train_gen(args.data, batch_size=args.data_size)
        dataset = [(d, 0) for d in dataset]  # add dummy labels
        num_train = int(args.data_size * .9)
        train_set, test_set = dataset[:num_train], dataset[num_train:]

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    return train_loader, test_loader


def get_loss(x):
    # batch of zeros for delta logpdf
    zero = torch.zeros(x.shape[0], 1).to(x)

    # backward to get z (standard normal)
    z, delta_logp = cnf(x, zero, reverse=True)

    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    # forward to get logpx
    # x_rec, logpx = cnf(z, logpz)
    # _logpx = logpz - delta_logp
    # print(torch.mean(torch.abs(_logpx - logpx)).item())
    # print(torch.mean(torch.abs(x - x_rec)).item())

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


if __name__ == '__main__':
    # get deivce
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    cvt = lambda x: x.type(torch.float32).to(device)

    # load dataset
    train_loader, test_loader = get_dataset(args)

    _odeint = integrate.odeint_adjoint if args.adjoint else integrate.odeint
    dims = list(map(int, args.dims.split(',')))
    dims = tuple([28**2] + dims) if args.data == "mnist" else tuple([2] + dims)

    cnf = models.CNF(dims=dims, T=args.time_length, odeint=_odeint)
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        cnf.load_state_dict(checkpt['state_dict'])
    cnf.to(device)

    optimizer = optim.Adam(cnf.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    end = time.time()
    best_loss = float('inf')
    itr = 0
    for epoch in range(1, args.num_epochs + 1):
        for _, (x, y) in enumerate(train_loader):
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            # cast data and move to device
            x = cvt(x)

            # compute loss
            loss = get_loss(x)
            loss.backward()

            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

            if itr % args.log_freq == 0:
                print('Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                    itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg)
                )

            itr += 1

        # compute test loss
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                print("validating...")
                losses = []
                for (x, y) in test_loader:
                    x = cvt(x)
                    loss = get_loss(x)
                    losses.append(loss.item())
                end = time.time()
                loss = np.mean(losses)
                print("Epoch {:04d} | Time {:.4f}, Loss {:.6f}".format(epoch, end - start, loss))
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': cnf.state_dict(),
                    }, os.path.join(args.save, 'checkpt.pth'))

        # visualize samples and density
        if epoch % args.viz_freq == 0:
            with torch.no_grad():
                if args.data == "mnist":
                    1/0
                else:
                    p_samples = toy_data.inf_train_gen(args.data, batch_size=10000)
                    plt.figure(figsize=(9, 3))
                    visualize_transform(
                        p_samples, torch.randn, standard_normal_logprob, cnf,
                        samples=True, device=device
                    )
                    fig_filename = os.path.join(args.save, 'figs', '{:04d}.jpg'.format(epoch))
                    utils.makedirs(os.path.dirname(fig_filename))
                    plt.savefig(fig_filename)

        end = time.time()