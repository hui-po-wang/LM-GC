cd ..

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NUM_SUBSAMPLE=10
DATASET='tinyimagenet' # cifar10 # mnist
ARCH="convnet" # vgg16 # resnet18 # vit
TYPE="grad"
COMPRESSOR="tinyllama" # llama2-7b # openllama3b
SEP="hex-none" # hex-space # hex-comma+space # iso # hex-semicolon
BATCHSIZE=4 # depending on your GPUs
BPG=4 # 8
for i in 1 2 3
do
  python -u compress.py -cfg settings/compression/cifar10-$SEP.yaml --compressor $COMPRESSOR --dataset tokenized_dataset \
    --data-path ./tokenized_datasets/$ARCH-$DATASET-$COMPRESSOR-$SEP-$NUM_SUBSAMPLE-$TYPE-$BPG-$i.pkl --batch-size $BATCHSIZE
done