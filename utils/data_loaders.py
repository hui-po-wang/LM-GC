"""Implements data loaders."""
from collections.abc import Iterator
import sys
sys.path.append('../')
import itertools, pickle, functools
import os, os.path
import random
import torch
import torch.nn as nn
import torchvision

import numpy as np
import torchvision.transforms as transforms

from transformers import AutoTokenizer

import constants
from utils.utils import BaseParser
from utils.misc import linear_quantization
from compressors import language_model, compressor
from typing import List

# number of bits represented by one symbol in a specific decoing scheme
SPLIT_VALUES = {
    'quater': 2,
    'oct': 3,
    'hex': 4,
}

SEP_VALUE = {
    None: '',
    'none': '',
    'comma': ',',
    'space': ' ',
    'comma+space': ', ',
    'semicolon': ';',
    '0x': '0x',
}

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

def _extract_image_patches(image: np.ndarray) -> Iterator[bytes]:
  h, w = constants.CHUNK_SHAPE_2D
  height, width = image.shape

  for row, col in itertools.product(range(height // h), range(width // w)):
    yield image[row * h : (row + 1) * h, col * w : (col + 1) * w].tobytes()

def _extract_parameter_patches(sample: bytes) -> Iterator[bytes]:
    patches = np.array_split(
      np.frombuffer(sample, dtype=np.uint8),
      range(
          constants.CHUNK_SIZE_BYTES,
          len(sample),
          constants.CHUNK_SIZE_BYTES,
      ),
    )
    if len(patches[-1]) != constants.CHUNK_SIZE_BYTES:
        # pad the array to have the same size
        current_size = len(patches[-1])
        padding_size = constants.CHUNK_SIZE_BYTES - current_size
        patches[-1] = np.pad(patches[-1], pad_width=(0, padding_size), mode='constant', constant_values=0)
    
    return map(lambda x: x.tobytes(), patches)

def _convert_bin_fp(data: bytes, precision=64) -> List[float]:
    converted_fp = np.frombuffer(data, dtype=np.float64 if precision == 64 else np.float32)
    return converted_fp

def _convert_fp_param(model: nn.Module, fp: float) -> None:
    # pointer: iterate through the entire dp and parse them to param
    pt = 0

    for _, p in model.named_parameters():
        len_p = p.numel()

        param_from_fp = torch.tensor(fp[pt:pt+len_p]).view(p.size()).to(model.device)
        p = param_from_fp
        
        pt = pt + len_p

def _serialize(data: bytes, decoding: str, bytes_per_group: int = 1) -> str:
    '''A function that convert bytes into a hex string list and then the formats that LLMs can understand.
    '''
    codec = decoding.split('-')[0]
    sep = decoding.split('-')[1] if '-' in decoding else None
    assert sep in SEP_VALUE.keys(), f'Unknown separator {sep}. This is typically controlled by the decoding argument.'
    sep = SEP_VALUE[sep]

    split = SPLIT_VALUES[codec] if codec in SPLIT_VALUES.keys() else 8
    
    # dump every byte as two hexi-decimal numbers and remove '0x' prefix
    hex_str = [hex(n)[2:].zfill(2) for n in data]

    # pre-group bytes according to the hyperparameter "bytes_per_group"
    hex_str = [''.join(hex_str[i:i+bytes_per_group]) for i in range(0, len(hex_str), bytes_per_group)]
    out = []

    # read one group, consisting of "bytes_per_group bytes" and represented by hex numbers
    for group in hex_str:
        concat_str = []
        num = int(group, 16)
        if codec == 'quater':
            pass
        elif codec == 'hex':
            num = hex(num)[2:].zfill(len(group))
        elif codec == 'iso':
            num = num.to_bytes(bytes_per_group, 'big') # orders should be taken care when loading the data
            num = num.decode('iso-8859-1')
        else:
            raise NotImplementedError(f'Unknown serialization method: {decoding}.')
        concat_str.append(num)
        num = ''.join(concat_str)

        out.append(num)

    out = sep.join(out)
    return out

def _deserialize(data: str, decoding: str, bytes_per_group: int = 1) -> bytes:
    '''A function that convert hex string list back to bytes.
    '''
    codec = decoding.split('-')[0]
    sep = decoding.split('-')[1] if '-' in decoding else None
    assert sep in SEP_VALUE.keys(), f'Unknown separator {sep}. This is typically controlled by the decoding argument.'
    sep = SEP_VALUE[sep]

    split = SPLIT_VALUES[codec] if codec in SPLIT_VALUES.keys() else 8

    def hex_char_to_bin(hex_char):
        return bin(int(hex_char, 16))[2:].zfill(4)

    # group data separated by the separators.
    grouped_data = data.split(sep)
    
    binary_string = ""
    for g in grouped_data:
        for char in g:
            # print(f'current string: {binary_string}, precoessing {char}')
            binary_string += hex_char_to_bin(char)

    byte_array = int(binary_string, 2).to_bytes((len(binary_string) + 7) // 8, byteorder='big')
    return byte_array


def get_image_iterator(
    args: BaseParser = None,
    serialization: bool = False,
    max_tokens: int = None,
    preprocess: str = None,
):
    trainset = IterImagePatchDataset(args, serialization)
    loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=1)
    return loader

def get_gradient_iterator(
    args: BaseParser = None,
    serialization: bool = False,
    max_tokens: int = None,
    preprocess: str = None,
    return_fp: bool=False
):
    trainset = IterGradientDataset(args, serialization, preprocess=preprocess, return_fp=return_fp)
    loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=1)
    return trainset.total_bytes, loader

def get_td_iterator(
    args: BaseParser = None,
    serialization: bool = False,
    max_tokens: int = None,
    preprocess: str = None,
):
    trainset = PreTokenizedDataset(args, max_tokens)
    loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=6, worker_init_fn=worker_init_fn)
    return trainset.total_bytes, loader

class IterImagePatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, serialization):
        self.args = args
        self.serialization = serialization

        transform_train = transforms.Compose([
            transforms.Grayscale(),
            # transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Grayscale(),
            # transforms.ToTensor(),
        ])
        
        if args.dataset == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(
                root=args.data_path, train=True, download=args.download, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR10(
                root=args.data_path, train=False, download=args.download, transform=transform_test)
            self.bytes_per_sample = 32 * 32 # only consider gray-scale images here
        elif args.dataset == 'imagenet':
            self.trainset = torchvision.datasets.ImageNet(
                root=args.data_path, train=True, download=args.download, transform=transform_train)
            self.testset = torchvision.datasets.ImageNet(
                root=args.data_path, train=False, download=args.download, transform=transform_test)
            self.bytes_per_sample = 256 * 256 # only consider gray-scale images here
        else:
            raise NotImplementedError(f'Unknown dataset: {args.dataset}')
        
    def __iter__(self):
        idx = 0

        for data in self.trainset:
            image, label = data
            if constants.UINT8_LOADING:
                image = np.array(image)
            else:
                image = image.squeeze().numpy()
            # print(type(image), image.shape)
            for patch in _extract_image_patches(image):
                num_bytes = len(patch)
                if idx == constants.NUM_CHUNKS:
                    return
                if self.serialization:
                    yield num_bytes, _serialize(patch, self.args.decoding, bytes_per_group=self.args.bytes_per_group)
                else:
                    yield num_bytes, patch
                idx += 1

class IterGradientDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, serialization, preprocess=None, return_fp=False):
        self.args = args
        self.serialization = serialization
        self.return_fp = return_fp
        self.ckpt_list = args.data_path
        if not isinstance(self.ckpt_list, list) and not isinstance(self.ckpt_list, np.ndarray):
            self.ckpt_list = [self.ckpt_list]

        preprocess_fn = []
        if preprocess is not None:
            preprocess = preprocess.split('+')
            for p in preprocess:
                print(f'Runing preprocessing {p} ...')
                if 'sparsification' in p:
                    strength = int(p.split('sparsification')[1]) / 100. # convert to percentage
                    preprocess_fn.append(functools.partial(self._sparsify, strength=strength))
                elif 'quantization' in p:
                    n_bits = int(p.split('quantization')[1])
                    preprocess_fn.append(functools.partial(self._quantize, n_bits=n_bits))
                else:
                    raise ValueError(f'Unknown preprocess instruction: {preprocess} specified in the dataloader.')

        self.length, self.total_bytes = 0, 0
        self.dset = []
        for ckpt in self.ckpt_list:
            state_dict = torch.load(ckpt, map_location=torch.device('cpu'))['state_dict']

            sample = []
            for k, v in state_dict.items():
                sample.append(v.numpy().flatten())
            sample = np.concatenate(sample)

            if len(preprocess_fn) > 0:
                for fn in preprocess_fn:
                    sample = fn(sample)
            
            print('===============> total bytes: ', len(sample.tobytes()))

            sample = sample.tobytes()
            self.length += int(np.ceil(len(sample)/constants.CHUNK_SIZE_BYTES))
            self.total_bytes += len(sample) if self.length < constants.NUM_CHUNKS else constants.NUM_CHUNKS * constants.CHUNK_SIZE_BYTES

            self.dset.append(sample)
            # sample: List[bytes]; len(sample) = #bytes contained in this ckpt
            
    
    def _add_gaussian_noise(self, data: np.ndarray, std: float=1):
        noise = np.random.normal(0, std, size=data.shape)
        return data + noise

    def _gradient_clipping(self, data: np.ndarray, strength: float=1):
        norm = np.linalg.norm(data)
        return data * (strength / norm) if norm > strength else data

    def _sparsify(self, data: np.ndarray, strength: float=0.25):
        # two way to implement this: set non-masked bits to zero (standard); only transmit masked bits (used by some existing works)
        # we use the latter here
        assert strength > 0 and strength <= 1
        mask = (np.random.uniform(0, 1, size=data.shape) < strength)
        return data[mask]

    def _is_power_of_two(self, n): 
        return (n & (n - 1)) == 0 and n != 0

    def _quantize(self, data: np.ndarray, n_bits: int=8):
        assert n_bits in [1, 8, 16]
        num_levels = 2 ** n_bits

        length = data.max() - data.min()
        if n_bits != 1:
            intervals = np.linspace(0, length, num_levels + 1)
            inds = np.digitize(data - data.min(), intervals)
        else:
            inds = data
            inds[inds>0] = 1
            inds[inds<=0] = 0

        if n_bits == 1:
            inds = np.packbits(inds.astype(bool))
        elif n_bits == 8:
            inds = inds.astype(np.uint8)
        elif n_bits == 16:
            inds = inds.astype(np.uint16)
        else:
            raise ValueError(f'Unsupported n_bits: {n_bits} for quantization.')

        return inds

    def __iter__(self):
        idx = 0
        for bid, s in enumerate(self.dset):
            if self.return_fp:
                step_size = constants.CHUNK_SIZE_BYTES // 4
                patches = np.array_split(
                    s,
                    range(
                        step_size,
                        len(s),
                        step_size,
                    ),
                )

                if len(patches[-1]) != step_size:
                    # pad the array to have the same size
                    current_size = len(patches[-1])
                    padding_size = step_size - current_size
                    patches[-1] = np.pad(patches[-1], pad_width=(0, padding_size), mode='constant', constant_values=0)

                for patch in patches:
                    yield sys.getsizeof(patch), patch
            else:
                for patch in _extract_parameter_patches(s):
                    num_bytes = len(patch)
                    if idx == constants.NUM_CHUNKS:
                        return
                    if self.serialization:
                        yield num_bytes, _serialize(patch, self.args.decoding, bytes_per_group=self.args.bytes_per_group)
                    else:
                        yield num_bytes, patch
                    idx += 1
    
    def __len__(self):
        return self.length

class PreTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, args, max_tokens=None):
        self.args = args
        with open(args.data_path, 'rb') as handle:
            loaded_data = pickle.load(handle)
            self.dset = loaded_data['tokens'].reshape(-1)
            self.total_bytes = loaded_data['total_bytes']

        if args.compressor in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model.MODEL_NAME_DICT[args.compressor])

            # we save one token quota for the bos_token which will be added later
            self.max_tokens = max_tokens - 1 if max_tokens is not None else 2048 - 1
            print(self.max_tokens)

            self.num_samples = int(np.ceil(len(self.dset)/(self.max_tokens + 1)))
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.max_tokens = constants.CHUNK_SIZE_BYTES
            self.num_samples = int(np.ceil(len(self.dset)/self.max_tokens))

    def _add_start_token(self, data: np.ndarray):
        # for some models like LLAMA, they append more than one tokens at the begining such as a space token
        # However, we ingore it here and only prepend a bos_token
        bos_token_id = self.tokenizer.bos_token_id
        return np.insert(data, 0, bos_token_id)

    def __getitem__(self, index):
        sample = self.dset[index*self.max_tokens:(index+1)*self.max_tokens]
        if self.args.compressor in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
            if len(sample) < self.max_tokens:
                padding = np.full((self.max_tokens, ), self.tokenizer.eos_token_id)
                padding[:len(sample)] = sample
                sample = padding
            sample = self._add_start_token(sample)
        return len(sample), sample

    def __len__(self):
        return self.num_samples 
            
GET_DATA_GENERATOR_FN_DICT = {
    'cifar10': get_image_iterator,
    'gradient': get_gradient_iterator,
    'tokenized_dataset': get_td_iterator,
}