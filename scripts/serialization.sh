cd ..

NUM_SUBSAMPLE=10
DATASET='tinyimagenet' # cifar10 # mnist
ARCH="convnet" # vgg16 # resnet18 # vit
TYPE="grad"
COMPRESSOR="tinyllama" # llama2-7b # openllama3b
SEP="hex-none" # hex-space # hex-comma+space # iso # hex-semicolon
BPG=4 # 8
for i in 1 2 3
do
  python -u tokenize_dataset.py --cfg settings/compression/cifar10-$SEP.yaml --data-path exps/$DATASET-$ARCH/0/grads/ --bytes-per-group $BPG \
  --compressor $COMPRESSOR --exhaustive-listing --num-subsample $NUM_SUBSAMPLE --output-name $ARCH-$DATASET-$COMPRESSOR-$SEP-$NUM_SUBSAMPLE-$TYPE-$BPG-$i 
done