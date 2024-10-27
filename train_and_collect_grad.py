import argparse, yaml, os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import OrderedDict

from utils.misc import get_network, prepare_dataset
from utils.utils import BaseParser

def parse_args():
    # specify the architecture, the frequency of gradient collection
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)
    parser.add_argument('-tag', '--tag', default=0, type=int)
    parser.add_argument('-seed', '--seed', default=None)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    
    parser.add_argument('-data-path', '--data-path', default='./', type=str)
    parser.add_argument('-download', '--download', action='store_true')

    parser.add_argument('-save_path', '--save_path', default='./saves', type=str)
    parser.add_argument('-grad-interval', '--grad-interval', default=200, type=int, help="The frequency of saving gradients. Counted by #batch steps.")
    parser.add_argument('-model-interval', '--model-interval', default=5, type=int, help="The frequency of saving models. Counted by #epoch.")

    parser.add_argument('-start-epoch', '--start-epoch', default=1, type=int, help="Load checkpoints, if sepcified.")

    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)

    args = BaseParser(args, settings)
    
    args.name = os.path.basename(args.cfg).split('.')[0]
    tag = args.tag
    args.exp_dir = os.path.join('exps/', args.name, f'{tag}')

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # save checkpoints
    args.model_dir = os.path.join(args.exp_dir, 'ckpts')
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # save gradients for compression experiments
    args.grad_dir = os.path.join(args.exp_dir, 'grads')
    if not os.path.exists(args.grad_dir):
        os.makedirs(args.grad_dir)

    # logs
    args.log_dir = os.path.join(args.exp_dir, 'logs')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args

def train(args, net, opt, crit, train_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        args.batch_step += 1

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        opt.zero_grad()

        outputs = net(inputs)
        loss = crit(outputs, targets)
        loss.backward()

        if args.batch_step % args.grad_interval == 0:
            grad_dict = {k: v.grad.detach().clone().cpu() for k, v in net.named_parameters()}
            state = {
                'state_dict': grad_dict,
                'batch_step': args.batch_step,
            }
            fname = os.path.join(args.grad_dir, f'grad_{args.batch_step:08d}.ckpt')
            torch.save(state, fname)
            
        opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return train_loss/len(train_loader), acc, correct, total

def test(args, net, crit, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = crit(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return test_loss/len(test_loader), acc, correct, total
    

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader, data_info = prepare_dataset(args)
    net = get_network(args.arch, data_info).to(args.device)

    crit = nn.CrossEntropyLoss()
    if 'vit' in args.arch:
        param_dict = {pn: p for pn, p in net.named_parameters()}
        parameters_decay, parameters_no_decay = net.separate_parameters()
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in parameters_decay], "weight_decay": 1e-1},
            {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
        ]
        opt = optim.AdamW(optim_groups, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                             steps_per_epoch=len(train_loader), epochs=200)
    else:
        opt = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    args.batch_step = 0 # just to avoid call-by-assign
    for epoch in tqdm(range(args.start_epoch, 200+1)):
        train_loss, train_acc, correct, total = train(args, net, opt, crit, train_loader)
        test_loss, test_acc, correct, total = test(args, net, crit, test_loader)
        print(f'At epoch {epoch}: Training accuracy {train_acc:.2f} and testing accuracy {test_acc:.2f}.')
        scheduler.step()

        if epoch % args.model_interval == 0:
            print(f'Saving models at epoch {epoch} ...')
            state = {
                'state_dict': net.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'epoch': epoch,
            }
            fname = os.path.join(args.model_dir, f'model_{epoch:03d}.ckpt')
            torch.save(state, fname)

if __name__ == "__main__":
    args = parse_args()
    main(args)