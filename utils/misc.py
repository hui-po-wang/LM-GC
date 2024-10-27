import pickle

import torch
import torchvision

import torchvision.transforms as transforms

import numpy as np

from utils.models.vgg import VGG
from utils.models.convnet import ConvNet
from utils.models.vit import ViT
from utils.models.resnet import resnet
from utils.tinyimagenet import TinyImageNet

def get_network(arch, data_info=None):
    if 'VGG' in arch:
        return VGG(arch)
    elif 'ConvNet' in arch:
        # default settings
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet(channel=data_info['channel'], num_classes=data_info['num_classes'], net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=data_info['im_size'])
    elif 'ViT' in arch:
        return ViT(data_info['num_classes'], data_info['im_size'][0])
    elif 'resnet' in arch:
        return resnet(arch, data_info['channel'], data_info['num_classes'])
    else:
        raise NotImplementedError(f'Unknown model architecture {arch}.')

def prepare_dataset(args):
    data_info = {}
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, download=args.download, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=False, download=args.download, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        data_info['classes'] = ('plane', 'car', 'bird', 'cat', 'deer', 
                                'dog', 'frog', 'horse', 'ship', 'truck')
        data_info['num_classes'] = 10
        data_info['channel'] = 3
        data_info['im_size'] = (32, 32)

    elif args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])

        trainset = torchvision.datasets.MNIST(
            root=args.data_path, train=True, download=args.download, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(
            root=args.data_path, train=False, download=args.download, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        data_info['classes'] = ('0', '1', '2', '3', '4', '5',
                                 '6', '7', '8', '9')
        data_info['num_classes'] = 10
        data_info['channel'] = 1
        data_info['im_size'] = (28, 28)
                                
    elif args.dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

        trainset = TinyImageNet(
            root=args.data_path, split='train', download=args.download, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=6)

        testset = TinyImageNet(
            root=args.data_path, split='val', download=args.download, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=6)

        # data_info['classes'] = 
        data_info['num_classes'] = 200
        data_info['channel'] = 3
        data_info['im_size'] = (32, 32)
    else:
        raise NotImplementedError(f'Unknown dataset type: {args.dataset}')
    
    return trainloader, testloader, data_info

# n_bits = 2, 4, 8
# unbiased: apply probabilistic unbiased quantization or not
# hadamard: apply random hadamard rotation or not
def linear_quantization(input, n_bits, unbiased=True, hadamard=True):
    quanti_level = 2 ** n_bits
    rand_diag = []

    if hadamard:
        input , rand_diag = hadamard_rotation(input)

    v_max = input.max()
    v_min = input.min()        
    output = input
    output = (output - v_min) / (v_max - v_min) * (quanti_level - 1)

    if unbiased:
        output = prob_quantization(output)
    else:
        output = output.round()

    #output = output.reshape(sz)

    return output, v_min, v_max, rand_diag, quanti_level

def hadamard_rotation(input):
    sz = input.shape
    sz1 = sz[0]
    sz2 = int(input.size / sz1)
    dim = 2 ** np.ceil(np.log2(sz1))
    hadamard_mat = hadamard(dim)
    if hadamard_mat.shape[0] != sz1:
        hadamard_mat = hadamard_mat[:sz1, :sz1]
    
    x = input.reshape(sz1, sz2)
    diag = np.random.uniform(0, 1, size=x.shape) < 0.5
    diag = diag * 2 - 1
    x = np.matmul(hadamard_mat, x) * diag
    x = x.reshape(sz)
    return x, diag

def prob_quantization(input):
    x = np.ceil(input)
    p = np.random.uniform(0, 1, size=x.shape)
    x = x - (p < x - input)
    return x

def hist_tokenized_dataset(data_path: str) -> np.ndarray:
    with open(data_path, 'rb') as handle:
        loaded_data = pickle.load(handle)
        dset = loaded_data['tokens'].reshape(-1)
        total_bytes = loaded_data['total_bytes']

        unique_elements, counts = np.unique(dset, return_counts=True)

        return unique_elements, counts