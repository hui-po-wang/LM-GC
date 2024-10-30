from collections.abc import Iterator
import functools, time
from typing import Callable

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torchac
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

import arithmetic_coder
import constants
from utils.utils import BaseParser
from utils import deepmind_utils

from multiprocessing import Pool

MODEL_NAME_DICT = {
  "gpt2": "gpt2",
  "openllama3b": "openlm-research/open_llama_3b_v2",
  "tinyllama3b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "mistral7b": "mistralai/Mistral-7B-v0.1",
  "llama2-7b": "meta-llama/Llama-2-7b-hf",
  "llama3-8b": "meta-llama/Meta-Llama-3-8B",
}
SKIP_LIST = ["openllama3b", "tinyllama3b", "tinyllama", "llama2-7b", "llama3-8b"]

def work(args):
  pid, cum_prob, input_ids = args
  byte_stream = torchac.encode_float_cdf(cum_prob, input_ids, check_input_bounds=False)
  return len(byte_stream)

class LanguageModelCompressor():
  def __init__(self, args: BaseParser):
    self.args = args
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(self.device)
    
    access_token = None
    assert access_token, 'Please enter the huggingface access token here if you want to use models like LLAMA.'

    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DICT[args.compressor], token=access_token)
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME_DICT[args.compressor],
      torch_dtype=torch.bfloat16,
      # device_map='auto',
      attn_implementation="flash_attention_2",
      token=access_token,
    )
    self.model.config.pad_token_id = self.model.config.eos_token_id
    self.model.to(self.device)
    self.model.eval()

  @torch.no_grad()
  def infer_prob(
      self,
      data: bytes,
      return_num_padded_bits: bool = False,
      use_slow_lossless_compression: bool = False,
  )  -> torch.Tensor:
    if self.args.dataset == 'tokenized_dataset':
      input_ids = data.to(self.device)#.view(1, -1)
      tokenized_data = {
            'input_ids': input_ids,
            'attention_mask': torch.ones(data.size()),
      }
    else:
      texified_data = data
      if not self.args.compressor in SKIP_LIST:
        texified_data = self.tokenizer.bos_token + texified_data

      print('length of texified data after adding eos:', len(texified_data))

      tokenized_data = self.tokenizer(texified_data, padding=False, return_tensors="pt").to(self.device)
      print(tokenized_data.keys())
      input_ids = tokenized_data.input_ids

    print('length of tokens', input_ids.shape)

    if use_slow_lossless_compression:
      # Compress the token at the position idx+1 using tokens_{t <= idx}
      """Remark!
      Due to the stochastic operations in huggingface and precision requirements by arithmetic coding, 
      one may use this option to compute the probability; however, in theory, they should not affect the result.

      TO-DO: provide a runnable example here.
      """
      for idx in range(len(tokenized_data.input_ids[0])-1):
        input_to_model = {
            'input_ids': tokenized_data.input_ids[:, :idx+1].view(1, -1),
            'attention_mask': tokenized_data.attention_mask[:, :idx+1].view(1, -1)
        }
        outputs = self.model(**input_to_model)

        logits = outputs.logits[:, idx, :] # next-word prob. of size # num_sentences x 
        pdf = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
        symbol = tokenized_data.input_ids[:, idx+1].item()
    else:
      # pay attention to the output length
      outputs = self.model(**tokenized_data)

      # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
      probs = outputs.logits[:, :-1, :] # num_sentences x next-word prob. of size (excl. the last word prediction)
      input_ids = input_ids[:, 1:] # number_sentences (1?) x max_legnth_chars-1 (excl. bos_token)

      probs = torch.log_softmax(probs, dim=-1).detach()
      probs = probs.exp()

      assert (input_ids < 65536).all()
      input_ids = input_ids.squeeze().detach().cpu().short()
      probs = probs.squeeze().detach().cpu()
    
    return input_ids, probs

  @torch.no_grad()
  def compress(
      self,
      data: bytes,
      return_num_padded_bits: bool = False,
      use_slow_lossless_compression: bool = False,
  ) -> bytes | tuple[bytes, int]:
      input_ids, probs = self.infer_prob(data, return_num_padded_bits=return_num_padded_bits, use_slow_lossless_compression=use_slow_lossless_compression)

      cum_prob = probs / torch.cumsum(probs, dim=-1)
      cum_prob = torch.cat([torch.zeros(cum_prob.size()[:-1]).unsqueeze(-1), cum_prob], dim=-1)

      byte_stream = torchac.encode_float_cdf(cum_prob, input_ids, check_input_bounds=False, needs_normalization=False)

      return byte_stream

  @torch.no_grad()
  def decompress(
      self,
      data: bytes,
      num_padded_bits: int = 0,
      uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
  ) -> bytes:
    """Decompresses the `data` using arithmetic coding and a pretrained model.

    See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

    Args:
      data: The data to be decompressed.
      num_padded_bits: The number of zeros added to the encoded bitstream in order
        to make it byte-decodeable (i.e., divisble by 8).
      uncompressed_length: The length of the original data stream (in bytes).

    Returns:
      The decompressed data.
    """
    data_iter = iter(deepmind_utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))
    # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
    # from the compressed input and returns `None` when the input is exhausted.
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
      try:
        return int(next(bit_sequence))
      except StopIteration:
        return None

    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )
    # We need a dummy token because the language model right-shifts the sequence
    # by one when computing the conditional probabilities. Concretely, at every
    # step, we need the `pdf` of the next token given all currently decompressed
    # tokens, but without a dummy token, the last `pdf` would be that of the last
    # already decompressed token. The value of the dummy token is irrelevant.
    input_ids = torch.empty((1, uncompressed_length), dtype=torch.long, device=self.device)
    input_ids[:, :] = self.tokenizer.pad_token_id
    attention_mask = torch.ones(input_ids.shape).to(self.device)

    # In our current implementation, we always begin with a <BOS> token.
    input_ids[0, 0] = self.tokenizer.bos_token_id

    for idx in range(0, uncompressed_length-1):
      input_to_model = {
          'input_ids': input_ids[:, :idx+1].view(1, -1),
          'attention_mask': attention_mask[:, :idx+1].view(1, -1)
      }
      outputs = self.model(**input_to_model)

      pdf = outputs.logits[:, idx, :] # next-word prob. of size # num_sentences x 
      pdf = torch.softmax(pdf, dim=-1).squeeze().detach().cpu().numpy()
      token = decoder.decode(
          deepmind_utils.normalize_pdf_for_arithmetic_coding(pdf)
      )
      input_ids[0, idx+1] = token

    input_ids = input_ids.cpu().detach().numpy()
    # Remove the dummy token and convert to bytes.
    return input_ids[0, :uncompressed_length]
