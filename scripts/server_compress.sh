#!/bin/bash
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_lmgc # Who pays for it?
#SBATCH --partition booster 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00 
#SBATCH -o output-%x.txt 
#SBATCH -e error-%x.txt
# Where does the code run?
# Required for legacy reasons
# How long?

source /p/home/jusers/wang34/juwels/hai_fedak/huipo/general/activate.sh # path to the environment
cd /p/home/jusers/wang34/juwels/hai_fedak/huipo/lmgc-to-release # path to the upper folder
LOG_DIR="scripts/test_compress"

if [ ! -d "$LOG_DIR" ]; then
  echo "$LOG_DIR does not exist."
  mkdir "$LOG_DIR"
else
  echo "$LOG_DIR does exist."
fi


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NUM_SUBSAMPLE=10
DATASET='tinyimagenet' # cifar10 # mnist
ARCH="convnet" # vgg16 # resnet18 # vit
TYPE="grad"
COMPRESSOR="tinyllama" # llama2-7b # openllama3b
SEP="hex-space" # hex-space # hex-comma+space # iso # hex-semicolon
BATCHSIZE=4 # number of windows processed everytime -- depending on your GPUs
BPG=4 # 8
INDEX=2
for i in 1 2 3
do
  srun --exclusive --gres=gpu:1 --cpus-per-task=6 --ntasks=1 -o "$LOG_DIR/out-$INDEX-$i.txt" -e "$LOG_DIR/err-$INDEX-$i.txt" \
  python -u compress.py -cfg settings/compression/cifar10-$SEP.yaml --compressor $COMPRESSOR --dataset tokenized_dataset \
    --data-path /p/home/jusers/wang34/juwels/hai_fedak/huipo/gradient_compressors/tokenized_datasets/$ARCH-$DATASET-$COMPRESSOR-$SEP-$NUM_SUBSAMPLE-$TYPE-$BPG-$i.pkl --batch-size $BATCHSIZE &
done

wait