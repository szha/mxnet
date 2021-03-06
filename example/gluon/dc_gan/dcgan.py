# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Generate MXNet implementation of Deep Convolutional Generative Adversarial Networks"""

import logging
from datetime import datetime
import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from inception_score import get_inception_score

mpl.use('Agg')


def fill_buf(buf, i, img, shape):
    """Reposition the images generated by the generator so that it can be saved as picture matrix.
    :param buf: the images metric
    :param i: index of each image
    :param img: images generated by generator once
    :param shape: each image`s shape
    :return: Adjust images for output
    """
    n = buf.shape[0]//shape[1]
    m = buf.shape[1]//shape[0]

    sx = (i%m)*shape[0]
    sy = (i//m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img


def visual(title, X, name):
    """Image visualization and preservation
    :param title: title
    :param X: images to visualized
    :param name: saved picture`s name
    :return:
    """
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X - np.min(X))*(255.0/(np.max(X) - np.min(X))), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = buff[:, :, ::-1]
    plt.imshow(buff)
    plt.title(title)
    plt.savefig(name)


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Train a DCgan model for image generation '
                                             'and then use inception_score to metric the result.')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use. options are cifar10 and mnist.')
parser.add_argument('--batch-size', type=int, default=64, help='input batch size, default is 64')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default is 100')
parser.add_argument('--ngf', type=int, default=64, help='the channel of each generator filter layer, default is 64.')
parser.add_argument('--ndf', type=int, default=64, help='the channel of each descriminator filter layer, '
                                                        'default is 64.')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for, default is 25.')
parser.add_argument('--niter', type=int, default=10, help='save generated images and inception_score per niter iters, '
                                                          'default is 100.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--check-point', default=True, help="save results at each epoch or not")
parser.add_argument('--inception_score', type=bool, default=True, help='To record the inception_score, '
                                                                       'default is True.')

opt = parser.parse_args()
print(opt)

logging.basicConfig(level=logging.DEBUG)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
niter = opt.niter
nc = 3
if opt.cuda:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()
batch_size = opt.batch_size
check_point = bool(opt.check_point)
outf = opt.outf
dataset = opt.dataset

if not os.path.exists(outf):
    os.makedirs(outf)


def transformer(data, label):
    """Get the translation of images"""
    # resize to 64x64
    data = mx.image.imresize(data, 64, 64)
    # transpose from (64, 64, 3) to (3, 64, 64)
    data = mx.nd.transpose(data, (2, 0, 1))
    # normalize to [-1, 1]
    data = data.astype(np.float32)/128 - 1
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = mx.nd.tile(data, (3, 1, 1))
    return data, label


# get dataset with the batch_size num each time
def get_dataset(dataset_name):
    """Load the dataset and split it to train/valid data

    :param dataset_name: string

    Returns:
    train_data: int array
        training dataset
    val_data: int array
        valid dataset
    """
    # mnist
    if dataset == "mnist":
        train_data = gluon.data.DataLoader(
            gluon.data.vision.MNIST('./data', train=True).transform(transformer),
            batch_size, shuffle=True, last_batch='discard')

        val_data = gluon.data.DataLoader(
            gluon.data.vision.MNIST('./data', train=False).transform(transformer),
            batch_size, shuffle=False)
    # cifar10
    elif dataset == "cifar10":
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10('./data', train=True).transform(transformer),
            batch_size, shuffle=True, last_batch='discard')

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10('./data', train=False).transform(transformer),
            batch_size, shuffle=False)

    return train_data, val_data


def get_netG():
    """Get net G"""
    # build the generator
    netG = nn.Sequential()
    with netG.name_scope():
        # input is Z, going into a convolution
        netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*8) x 4 x 4
        netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*4) x 8 x 8
        netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf*2) x 16 x 16
        netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
        netG.add(nn.BatchNorm())
        netG.add(nn.Activation('relu'))
        # state size. (ngf) x 32 x 32
        netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
        netG.add(nn.Activation('tanh'))
        # state size. (nc) x 64 x 64

    return netG


def get_netD():
    """Get the netD"""
    # build the discriminator
    netD = nn.Sequential()
    with netD.name_scope():
        # input is (nc) x 64 x 64
        netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf) x 32 x 32
        netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
        netD.add(nn.BatchNorm())
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf*2) x 16 x 16
        netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
        netD.add(nn.BatchNorm())
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf*4) x 8 x 8
        netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
        netD.add(nn.BatchNorm())
        netD.add(nn.LeakyReLU(0.2))
        # state size. (ndf*8) x 4 x 4
        netD.add(nn.Conv2D(2, 4, 1, 0, use_bias=False))
        # state size. 2 x 1 x 1

    return netD


def get_configurations(netG, netD):
    """Get configurations for net"""
    # loss
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # initialize the generator and the discriminator
    netG.initialize(mx.init.Normal(0.02), ctx=ctx)
    netD.initialize(mx.init.Normal(0.02), ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': opt.lr, 'beta1': opt.beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': opt.lr, 'beta1': opt.beta1})

    return loss, trainerG, trainerD


def ins_save(inception_score):
    # draw the inception_score curve
    length = len(inception_score)
    x = np.arange(0, length)
    plt.figure(figsize=(8.0, 6.0))
    plt.plot(x, inception_score)
    plt.xlabel("iter/100")
    plt.ylabel("inception_score")
    plt.savefig("inception_score.png")


# main function
def main():
    """Entry point to dcgan"""
    print("|------- new changes!!!!!!!!!")
    # to get the dataset and net configuration
    train_data, val_data = get_dataset(dataset)
    netG = get_netG()
    netD = get_netD()
    loss, trainerG, trainerD = get_configurations(netG, netD)

    # set labels
    real_label = mx.nd.ones((opt.batch_size,), ctx=ctx)
    fake_label = mx.nd.zeros((opt.batch_size,), ctx=ctx)

    metric = mx.gluon.metric.Accuracy()
    print('Training... ')
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    iter = 0

    # to metric the network
    loss_d = []
    loss_g = []
    inception_score = []

    for epoch in range(opt.nepoch):
        tic = time.time()
        btic = time.time()
        for data, _ in train_data:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real_t
            data = data.as_in_context(ctx)
            noise = mx.nd.random.normal(0, 1, shape=(opt.batch_size, nz, 1, 1), ctx=ctx)

            with autograd.record():
                output = netD(data)
                # reshape output from (opt.batch_size, 2, 1, 1) to (opt.batch_size, 2)
                output = output.reshape((opt.batch_size, 2))
                errD_real = loss(output, real_label)

            metric.update([real_label, ], [output, ])

            with autograd.record():
                fake = netG(noise)
                output = netD(fake.detach())
                output = output.reshape((opt.batch_size, 2))
                errD_fake = loss(output, fake_label)
                errD = errD_real + errD_fake

            errD.backward()
            metric.update([fake_label,], [output,])

            trainerD.step(opt.batch_size)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            with autograd.record():
                output = netD(fake)
                output = output.reshape((-1, 2))
                errG = loss(output, real_label)

            errG.backward()

            trainerG.step(opt.batch_size)

            name, acc = metric.get()
            logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                         , mx.nd.mean(errD).asscalar(), mx.nd.mean(errG).asscalar(), acc, iter, epoch)
            if iter % niter == 0:
                visual('gout', fake.asnumpy(), name=os.path.join(outf, 'fake_img_iter_%d.png' % iter))
                visual('data', data.asnumpy(), name=os.path.join(outf, 'real_img_iter_%d.png' % iter))
                # record the metric data
                loss_d.append(errD)
                loss_g.append(errG)
                if opt.inception_score:
                    score, _ = get_inception_score(fake)
                    inception_score.append(score)

            iter = iter + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f', epoch, name, acc)
        logging.info('time: %f', time.time() - tic)

        # save check_point
        if check_point:
            netG.save_parameters(os.path.join(outf, 'generator_epoch_%d.params' %epoch))
            netD.save_parameters(os.path.join(outf, 'discriminator_epoch_%d.params' % epoch))

    # save parameter
    netG.save_parameters(os.path.join(outf, 'generator.params'))
    netD.save_parameters(os.path.join(outf, 'discriminator.params'))

    # visualization the inception_score as a picture
    if opt.inception_score:
        ins_save(inception_score)


if __name__ == '__main__':
    if opt.inception_score:
        print("Use inception_score to metric this DCgan model, the reusult is save as a picture "
              "named \"inception_score.png\"!")
    main()
