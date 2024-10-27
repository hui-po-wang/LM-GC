NUM_CHUNKS = 10000000
CHUNK_SIZE_BYTES = 512 # always read this amount of data, regardless of data type
CHUNK_SHAPE_2D = (16, 32)
UINT8_LOADING = True # only useful for images

# Base 2 means that the coder writes bits.
ARITHMETIC_CODER_BASE = 2
# Precision 32 implies 32 bit arithmetic.
ARITHMETIC_CODER_PRECISION = 32