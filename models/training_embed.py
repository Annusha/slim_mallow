#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from os.path import join
import time

from utils.arg_pars import opt
from utils.logging_setup import logger
from models.dataset_torch import load_data, load_mnist
from utils.utils import AverageMeter, join_data, adjust_lr
from utils.visualization import Visual
from utils.utils import dir_check
from models.rank import Embedding, RankLoss


def training(train_loader, epochs, n_subact=0, save=True, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    logger.debug('create model')
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    try:
        model = kwargs['model']
        loss = kwargs['loss']
        optimizer = kwargs['optimizer']
    except KeyError:
        model = Embedding(embed_dim=opt.embed_dim,
                          feature_dim=opt.feature_dim,
                          n_subact=n_subact).cuda()

        loss = RankLoss(margin=0.2).cuda()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    cudnn.benchmark = True

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    vis = Visual()
    best_acc = -1
    _lr = opt.lr

    logger.debug('epochs: %s', epochs)
    loss_previous = np.inf
    for epoch in range(epochs):
        model.cuda()
        model.train()

        logger.debug('Epoch # %d' % epoch)
        if opt.lr_adj:
            # if epoch in [int(epochs * 0.3), int(epochs * 0.7)]:
            # if epoch in [int(epochs * 0.5)]:
            if epoch % 30 == 0 and epoch > 0:
                _lr = adjust_lr(optimizer, _lr)
                logger.debug('lr: %f' % _lr)
        end = time.time()
        for i, (input, k, _) in enumerate(train_loader):
            # TODO: not sure that it's necessary
            data_time.update(time.time() - end)
            input = input.float().cuda(non_blocking=True)
            k = k.float().cuda()
            output = model(input)
            loss_values = loss(output, k)
            losses.update(loss_values.item(), input.size(0))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        logger.debug('loss: %f' % losses.avg)
        losses.reset()

    if save:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(join(opt.dataset_root, 'models'))
        dir_check(join(opt.dataset_root, 'models', kwargs['name']))
        torch.save(save_dict, join(opt.dataset_root, 'models', kwargs['name'],
                                   '%s.pth.tar' % opt.log_str))
    return model


def visualization2d(train_loader, model, epoch, vis, resume=False):
    """To get understanding how embedding separate space using whatever loss"""
    if resume:
        logger.debug('Load the model for epoch %d' % epoch)
        model.load_state_dict(load_model(epoch))
    else:
        model.cpu()

    model.eval()

    logger.debug('Evaluation')
    with torch.no_grad():
        anchors = model.anchors().detach().numpy()

        vis.data = anchors
        vis.labels = np.arange(anchors.shape[0])
        vis.size = 5
        for i, (input, k, _) in enumerate(train_loader):
            input = input.float()
            k = k.numpy()
            vis.labels = np.argmax(k, axis=1)

            output = model.embedded(input).cpu().numpy()

            vis.data = output
            if i == int(len(train_loader) * 0.5):
                break
    logger.debug('Apply data reduction')
    vis.fit_data()
    vis.plot(epoch=epoch)
    vis.reset()


def accuracy(train_loader, model, epoch, best_acc, resume=False, idx2name=None):
    """Calculate accuracy of trained embedding either just trained or with
    pretrained model"""
    if resume:
        logger.debug('Load the model for epoch %d' % epoch)
        model.load_state_dict(load_model(epoch))
    else:
        model.cpu()

    model.eval()
    acc = AverageMeter()

    logger.debug('Evaluation')
    with torch.no_grad():
        anchors = model.anchors().detach().numpy()

        video_save_feat = None
        name_cur = None
        for i, (input, k, name) in enumerate(train_loader):
            input = input.float()
            k = k.numpy()
            k = np.argmax(k, axis=1)

            output = model.embedded(input).cpu().numpy()
            if opt.save:
                name = name.numpy()
                name_cur = name[0] if name_cur is None else name_cur
                for idx, f in enumerate(output):
                    if name_cur == int(name[idx]):
                        video_save_feat = join_data(video_save_feat, f, np.vstack)
                    else:
                        np.savetxt(join(opt.data, 'embed', '%d_%d_%s_' %
                                        (opt.embed_dim, opt.data_type, str(opt.lr))
                                        + idx2name[name_cur]),
                                   video_save_feat)
                        video_save_feat = join_data(None, f, np.vstack)
                        name_cur = int(name[idx])
            dists = -2 * np.dot(output, anchors.T) + np.sum(anchors ** 2, axis=1) \
                    + np.sum(output ** 2, axis=1)[:, np.newaxis]

            dist = np.sum(np.argmin(dists, axis=1) == k, dtype=float) / input.size(0)
            acc.update(dist, input.size(0))
            if i % 100 == 0 and i:
                logger.debug('Iter: [{0}/{1}]\t'
                             'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                    i, len(train_loader), acc=acc))
        if opt.save_feat:
            np.savetxt(join(opt.data, 'embed',
                            '%d_%d_%s_' % (opt.embed_dim, opt.data_type, str(opt.lr))
                            + idx2name[name_cur]), video_save_feat)
            np.savetxt(join(opt.data, 'embed', 'anchors_%s_%d_%d_%s'
                            % (opt.subaction, opt.embed_dim, opt.data_type, str(opt.lr))),
                       anchors)
        if best_acc < acc.avg:
            best_acc = acc.avg
            logger.debug('Accuracy {acc.val:.4f} ({acc.avg:.4f})\t(best:{0:.4f})'
                         .format(best_acc, acc=acc))
    return best_acc


def load_model(name=None):

    if opt.resume_str:
        subaction = opt.subaction.split('_')[0]
        resume_str = opt.resume_str % subaction
        # resume_str = opt.resume_str
    else:
        resume_str = opt.log_str
    checkpoint = torch.load(join(opt.dataset_root, 'models', name,
                                 '%s.pth.tar' % resume_str))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s.pth.tar' % resume_str)
    return checkpoint


def resume(train_loader, epoch, n_subact, feature_dim, idx2name):
    """Resume to calculate accuracy with pretrained models"""
    model = Embedding(embed_dim=opt.embed_dim,
                      feature_dim=feature_dim,
                      n_subact=n_subact)
    vis = Visual(mode='tsne', full=True)

    accuracy(train_loader, model, epoch, best_acc=0, resume=True,
             idx2name=idx2name)
    # visualization2d(train_loader, model, epoch, vis, mnist=mnist, resume=True)


def pipeline():
    idx2name = [0]
    dataloader, n_subact = load_data(opt.data, opt.end, subaction=opt.subaction,
                                     names=idx2name)
    feature_dim = opt.feature_dim

    for lr in [1e-5, 1e-6, 1e-7]:
        opt.lr = lr
        for i in [0, opt.epochs - 1]:
            opt.resume = i

            if opt.resume:
                resume(dataloader, epoch=opt.resume, n_subact=n_subact, feature_dim=feature_dim, idx2name=idx2name[0])
            else:
                training(dataloader, opt.epochs, n_subact)


if __name__=='__main__':
    pipeline()
