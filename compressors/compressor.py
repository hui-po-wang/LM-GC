# Copyright 2024 DeepMind Technologies Limited
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

"""Defines the compressor interface."""
import os

import functools
import gzip
import lzma
import fpzip
from typing import Mapping, Protocol

from compressors import flac
from compressors import language_model
from compressors import png


class Compressor(Protocol):

  def __call__(self, data: bytes, *args, **kwargs) -> bytes | tuple[bytes, int]:
    """Returns the compressed version of `data`, with optional padded bits."""


COMPRESSOR_TYPES = {
    'classical': ['flac', 'gzip', 'lzma', 'png', 'fpzip'],
    'arithmetic_coding': ['gpt2', 'openllama3b', 'tinyllama3b', 'tinyllama', 'mistral7b', 'llama2-7b', 'llama3-8b'],
}

COMPRESS_FN_DICT: Mapping[str, Compressor] = {
    'flac': flac.compress,
    'gzip': functools.partial(gzip.compress, compresslevel=9),
    'gpt2': language_model.LanguageModelCompressor,
    'openllama3b': language_model.LanguageModelCompressor,
    'tinyllama3b': language_model.LanguageModelCompressor,
    'tinyllama': language_model.LanguageModelCompressor,
    'mistral7b':  language_model.LanguageModelCompressor,
    'llama2-7b':  language_model.LanguageModelCompressor,
    'llama3-8b':  language_model.LanguageModelCompressor,
    'lzma': lzma.compress,
    'png': png.compress,
    'fpzip': functools.partial(fpzip.compress, precision=0, order='C'),
}