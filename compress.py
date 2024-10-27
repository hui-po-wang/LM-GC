# Copyright 2024 CISPA Helmholtz Center for Information Security Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# import sys
# sys.path.append('../gradient-compressors')
import argparse, os
os.environ['HF_HOME'] = './cache'

import functools
import time
import logging, yaml

from collections.abc import Generator
from typing import Callable

import numpy as np

import torch

import tqdm
import constants

from utils import data_loaders
from utils import deepmind_utils
from utils.utils import BaseParser
from compressors import compressor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)

    parser.add_argument('-data-path', '--data-path', default='./', required=True, type=str)
    parser.add_argument('-exhaustive-listing', '--exhaustive-listing', action='store_true', help='If read all of the checkpoints in the data path.')
    parser.add_argument('-num-subsample', '--num-subsample', type=int, default=1)

    parser.add_argument('-download', '--download', action='store_true')
    
    parser.add_argument('-use_mask', '--use_mask', action='store_true', help='Applying mask functions, particularly for decoding images into ASCII.')
    parser.add_argument('-use_slow_compression', '--use_slow_compression', action='store_true')

    parser.add_argument('--dataset', '-dataset', default='tokenized_dataset', type=str, help='Indicatge what kind of data to compress.')
    parser.add_argument('--compressor', '-compressor', default='gpt2', type=str, help='What kind of compressor to use.')
    
    parser.add_argument('--bytes-per-group', '-bytes-per-group', default=None, type=int, help='Specify after how many bytes a separator will be added.')
    parser.add_argument('-batch-size', '--batch-size', type=int, default=32)
    parser.add_argument('-max-tokens', '--max-tokens', type=int, default=2048)
    parser.add_argument('-preprocess', '--preprocess', type=str, default=None)

    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)

    args = BaseParser(args, settings)

    # handle bytes_per_group
    if args.bytes_per_group is None:
        args.bytes_per_group = 4 if args.dataset == 'tokenized_data' or args.dataset == 'gradient' else 1
    
    if args.exhaustive_listing:
        paths = np.array([os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, f))])
        args.sample_index = np.random.choice(len(paths), args.num_subsample, replace=False)
        print(args.sample_index)
        args.data_path = paths[args.sample_index]

    print(args)
    print('chunck size: ', constants.CHUNK_SIZE_BYTES)
    print('num_chuncks: ',constants.NUM_CHUNKS)
    return args

def evaluate_compressor_chunked(
    args: BaseParser,
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
    count_header_only_once: bool = True,
    mask_fn: Callable[[bytes], tuple[bytes, int]] | None = None,
    use_tqdm: bool = True,
) -> tuple[float, float]:
    """Evaluates the compressor on the chunked dataset.

    Args:
        compress_fn: The function that evaluates data.
        get_data_generator_fn: The function that creates a data generator.
        num_chunks: The number of chunks to consider
        count_header_only_once: Whether to count the header as part of the
        compressed output only once for the whole dataset or for every chunk
        individually.
        mask_fn: The function that masks the data in case the compressor cannot
        handle all possible byte values (e.g., language models can only process
        ASCII-decodable data).
        use_tqdm: Whether to use a progress bar or not.

    Returns:
        The compression rate and the total running time.
    """
    num_missed_bits = running_time = raw_length = compressed_length = num_samples = 0

    raw_length, data_generator = get_data_generator_fn()
    print(f'Data to compress has size {raw_length} bytes.')

    if args.dataset == 'tokenized_dataset':
        for num_bytes, data in tqdm.tqdm(data_generator):
            num_samples += len(data)
            if mask_fn is not None:
                d, missed_bits = mask_fn(data)
                num_missed_bits += missed_bits
            
            t0 = time.perf_counter()
            compressed_data = compress_fn(data, use_slow_lossless_compression=args.use_slow_compression)
            t1 = time.perf_counter()

            running_time += t1 - t0
            compressed_length += len(compressed_data)
    else:
        for num_bytes, data in tqdm.tqdm(data_generator):
            num_samples += len(data)
            for d_size, d in zip(num_bytes, data):
                if mask_fn is not None:
                    d, missed_bits = mask_fn(d)
                    num_missed_bits += missed_bits

                if isinstance(d, torch.Tensor):
                    d = d.numpy()
                    
                t0 = time.perf_counter()
                compressed_data = compress_fn(d)
                t1 = time.perf_counter()

                running_time += t1 - t0
                compressed_length += len(compressed_data)

        # raw_length += constants.CHUNK_SIZE_BYTES * num_chunks

    # Since language models are trained on ASCII strings, they cannot handle all
    # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
    # `num_missed_bits` of the most significant bits. However, this means that we
    # are effectively only compressing `num_bits - num_missed_bits` bits, so we
    # rescale the `compressed_length` to account for this.
    if mask_fn is not None:
        num_bits = 8 * num_samples * constants.CHUNK_SIZE_BYTES
        compressed_length *= num_bits / (num_bits - num_missed_bits)

    # We only count the header once for classical compressors.
    # if count_header_only_once:
    #     header_length = len(compress_fn((0).to_bytes(1, 'little')))
    #     compressed_length -= header_length * (num_samples - 1)

    return compressed_length / raw_length, running_time


def evaluate_compressor_unchunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
) -> tuple[float, float]:
    """Evaluates the compressor on the unchunked dataset.

    Args:
        compress_fn: The function that compresses data.
        get_data_generator_fn: The function that creates a data generator.
        num_chunks: The number of chunks to consider.

    Returns:
        The compression rate and the total running time.
    """
    all_data = None

    raw_length, data_generator = get_data_generator_fn()
    print(f'Data to compress has size {raw_length} bytes.')

    with tqdm.tqdm(total=constants.NUM_CHUNKS) as pbar:
        for num_bytes, data in data_generator:
            for d in data:
                if all_data is None:
                    all_data = bytearray() if isinstance(d, bytes) else []
                
                if isinstance(d, bytes):
                    all_data += d
                else:
                    d = d.numpy()
                    all_data.append(d)

                pbar.update(1)

    all_data = bytes(all_data) if isinstance(all_data[0], bytes) else np.concatenate(all_data)

    t0 = time.perf_counter()
    compressed_data = compress_fn(all_data)
    t1 = time.perf_counter()

    return len(compressed_data) / raw_length, t1 - t0


def main(args) -> None:
    print('start')
    compress_fn = compressor.COMPRESS_FN_DICT[args.compressor]

    if args.compressor in compressor.COMPRESSOR_TYPES['classical']:
        get_data_generator_fn = functools.partial(
            data_loaders.GET_DATA_GENERATOR_FN_DICT[args.dataset],
            args=args,
            serialization=False,
            preprocess=args.preprocess,
            return_fp=False if args.compressor != 'fpzip' else True
        )
        unchunked_rate, unchunked_time = evaluate_compressor_unchunked(
            compress_fn=compress_fn,
            get_data_generator_fn=get_data_generator_fn,
            num_chunks=constants.NUM_CHUNKS,
        )
        chunked_rate, chunked_time = evaluate_compressor_chunked(
            args=args,
            compress_fn=compress_fn,
            get_data_generator_fn=get_data_generator_fn,
            num_chunks=constants.NUM_CHUNKS,
            count_header_only_once=True,
            mask_fn=None,
        )
        print(
            f'Unchunked: {100 * unchunked_rate:.2f} [{unchunked_time:.1f}s]'
        )
        print(f'Chunked: {100 * chunked_rate:.2f} [{chunked_time:.1f}s]')

    elif args.compressor in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
        get_data_generator_fn = functools.partial(
            data_loaders.GET_DATA_GENERATOR_FN_DICT[args.dataset],
            args=args,
            serialization=True,
            max_tokens=args.max_tokens,
            preprocess=args.preprocess,
        )
        model = compress_fn(args)
        if args.use_mask:
            # To compress bytes data, we convert it first to ASCII.
            if args.dataset == 'enwik9':
                # For Enwik9, some characters are UTF-8 but not ASCII, so we still need
                # to do the conversion.
                mask_fn = deepmind_utils.zero_most_significant_bit_if_not_ascii_decodable
            else:
                mask_fn = deepmind_utils.right_shift_bytes_by_one
        else:
            mask_fn = None

        chunked_rate, chunked_time = evaluate_compressor_chunked(
            args = args,
            compress_fn=model.compress,
            get_data_generator_fn=get_data_generator_fn,
            num_chunks=constants.NUM_CHUNKS,
            count_header_only_once=False,
            mask_fn=mask_fn,
        )
        print(f'Chunked: {100 * chunked_rate:.2f} [{chunked_time:.1f}s]')
    else:
        raise NotImplementedError(f'Unknown codec {args.compressor}.')
    
if __name__ == "__main__":
    args = parse_args()
    main(args)