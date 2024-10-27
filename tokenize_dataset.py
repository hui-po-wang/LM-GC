import argparse, yaml
import functools, pickle
import os
os.environ['HF_HOME'] = './cache/'

import constants
import numpy as np

from transformers import AutoTokenizer

from tqdm import tqdm
from utils import data_loaders
from utils.utils import BaseParser
from compressors import language_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)

    parser.add_argument('-data-path', '--data-path', default='./', required=True, type=str)
    parser.add_argument('-exhaustive-listing', '--exhaustive-listing', action='store_true', help='If read all of the checkpoints in the data path.')
    parser.add_argument('-num-subsample', '--num-subsample', required=True, type=int)

    parser.add_argument('-download', '--download', action='store_true')
    
    parser.add_argument('-use_mask', '--use_mask', action='store_true', help='Applying mask functions, particularly for decoding images into ASCII.')
    
    parser.add_argument('--dataset', '-dataset', default='gradient', type=str, help='Indicatge what kind of data to compress.')

    parser.add_argument('--output-name', '-output-name', required=True, help='Name of the output pre-tokenized dataset.')

    parser.add_argument('--compressor', '-compressor', default='tinyllama3b', type=str, help='What kind of compressor to use.')
    parser.add_argument('--verbose', '-verbose', action='store_true', help='Print first few tokens for debugging.')

    parser.add_argument('--bytes-per-group', '-bytes-per-group', default=None, type=int, help='Specify after how many bytes a separator will be added.')
    parser.add_argument('-batch-size', '--batch-size', type=int, default=32)
    parser.add_argument('-preprocess', '--preprocess', type=str, default=None)

    parser.add_argument('-noise-level', '--noise-level', default=None, type=float, help='Standard deviation of gaussian noise to add. Useful only when preprocess "gaussian" is enabled.')
    parser.add_argument('-clipping-bound', '--clipping-bound', default=None, type=float, help='Clipping bound. Useful only when preprocess "clipping" is enabled.')

    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)

    args = BaseParser(args, settings)

    args.output_dir = './tokenized_datasets'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # arch-cifar-lm-serializeation-step-[model|grad]-tag.pkl
    args.output_name = os.path.join(args.output_dir, args.output_name+'.pkl')

    # handle bytes_per_group
    if args.bytes_per_group is None:
        args.bytes_per_group = 4 if args.dataset == 'tokenized_data' or args.dataset == 'gradient' else 1

    print(args)
    print('chunck size: ', constants.CHUNK_SIZE_BYTES)
    print('num_chuncks: ',constants.NUM_CHUNKS)

    if args.exhaustive_listing:
        paths = np.array([os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, f))])
        args.sample_index = np.random.choice(len(paths), args.num_subsample, replace=False)
        args.data_path = paths[args.sample_index]
        
    # print(args.data_path)
    
    return args

def main(args):
    _, data_generator = data_loaders.GET_DATA_GENERATOR_FN_DICT[args.dataset](args=args, serialization=True, preprocess=args.preprocess)
    tokenizer = AutoTokenizer.from_pretrained(language_model.MODEL_NAME_DICT[args.compressor])
    tokenizer.pad_token = tokenizer.eos_token

    skip_tokens = {
        'tinyllama3b': [1, 29871],
        'tinyllama': [1, 29871],
        'openllama3b': [1, 29871],
        'gpt2': [],
        'llama3-8b': [128000],
        'llama2-7b': [1, 29871],
    }
    # tokenize -> remove strat token if using LLAMA; sometimes tokenizer also add a '' token with id 29871 to the data
    concat_tokens = []

    assert args.compressor in skip_tokens.keys()

    print(f'Ready to tokenize {len(data_generator)} * 32 samples.')
    print(f'Each sample consists of {constants.CHUNK_SIZE_BYTES} bytes.')
    print(f'The preprocess will be hanlded by {args.compressor}\'s tokenizer.')

    num_sample = 0
    for num_bytes, data in tqdm(data_generator):
        num_sample += len(data)
        for d_size, d in zip(num_bytes, data):
            tokenized_data = tokenizer(d, padding=False, return_tensors="pt")#.view(-1) # size: num_tokens
            tokenized_data = tokenized_data.input_ids.view(-1)
            if args.verbose:
                print(tokenized_data[:10])
                print(f'\t{d[:20]}')
                print(f'\t{[tokenizer.decode(t) for t in tokenized_data[:10]]}')

            # while loop pop bos and additional space tokens
            started_index = 0
            while tokenized_data[started_index] in skip_tokens[args.compressor]:
                started_index += 1
            
            tokenized_data = tokenized_data[started_index:]
            concat_tokens.append(tokenized_data.cpu().detach().numpy())

            if args.verbose:
                print(tokenized_data[:10])
                print(f'\t{[tokenizer.decode(t) for t in tokenized_data[:10]]}')

    concat_tokens = np.concatenate(concat_tokens)
    print(f'Processed dataset contains {num_sample} samples with {len(concat_tokens)} tokens, containing {num_sample * constants.CHUNK_SIZE_BYTES} byte'
        f' ~= {num_sample * constants.CHUNK_SIZE_BYTES/1024/1024:.2f} MB.')

    with open(args.output_name, 'wb') as handle:
        pickle.dump({'tokens': concat_tokens, 'sample_index': args.sample_index, 'total_bytes': num_sample * constants.CHUNK_SIZE_BYTES}, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()
    main(args)