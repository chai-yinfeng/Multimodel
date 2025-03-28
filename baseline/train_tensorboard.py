import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
import data
import model_attention
import utils


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0, writer=None):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax(dim=-1).cuda()
    for v, q, a, idx, q_len in tq:
        if train:
            v = v.cuda(non_blocking=True)
            q = q.cuda(non_blocking=True)
            a = a.cuda(non_blocking=True)
            q_len = q_len.cuda(non_blocking=True)
        else:
            with torch.no_grad():
                v = v.cuda(non_blocking=True)
                q = q.cuda(non_blocking=True)
                a = a.cuda(non_blocking=True)
                q_len = q_len.cuda(non_blocking=True)

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录训练损失和准确率到 TensorBoard
            if writer:
                writer.add_scalar('Train/Loss', loss.item(), total_iterations)
                writer.add_scalar('Train/Accuracy', acc.mean().item(), total_iterations)

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())
        # acc_tracker.append(acc.mean())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        # 记录验证损失和准确率到 TensorBoard
        if writer:
            writer.add_scalar('Validation/Loss', loss_tracker.mean.value, epoch)
            writer.add_scalar('Validation/Accuracy', acc_tracker.mean.value, epoch)

        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model_attention.Net(train_loader.dataset.num_tokens)).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    # 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('logs', name))

    for i in range(config.epochs):
        # 训练和验证过程记录到Tensorboard
        _ = run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i, writer=writer)
        r = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i, writer=writer)

        # results = {
        #     'name': name,
        #     'tracker': tracker.to_dict(),
        #     'config': config_as_dict,
        #     'weights': net.state_dict(),
        #     'eval': {
        #         'answers': r[0],
        #         'accuracies': r[1],
        #         'idx': r[2],
        #     },
        #     'vocab': train_loader.dataset.vocab,
        # }
        # torch.save(results, target_name)

    # 关闭 TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
