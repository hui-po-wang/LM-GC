#!/bin/bash
#SBATCH --nodes=1 # How many nodes? 
#SBATCH -A hai_lmgc # Who pays for it?
#SBATCH --partition develbooster 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00 
#SBATCH -o output-%x.txt 
#SBATCH -e error-%x.txt
# Where does the code run?
# Required for legacy reasons
# How long?

source /p/home/jusers/wang34/juwels/hai_fedak/huipo/general/activate.sh # path to the environment
cd /p/home/jusers/wang34/juwels/hai_fedak/huipo/lmgc-to-release # path to the upper folder
LOG_DIR="scripts/test_collect"

if [ ! -d "$LOG_DIR" ]; then
  echo "$LOG_DIR does not exist."
  mkdir "$LOG_DIR"
else
  echo "$LOG_DIR does exist."
fi

DATASET='tinyimagenet'
ARCH="convnet"
INDEX=1
for i in 0 1 2
do
    srun --exclusive --gres=gpu:1 --cpus-per-task=6 --ntasks=1 -o "$LOG_DIR/out-$INDEX-$i.txt" -e "$LOG_DIR/err-$INDEX-$i.txt" \
    python -u train_and_collect_grad.py -cfg settings/gradient_collection/$DATASET-$ARCH.yaml --tag $i --grad-interval 400 --download &
done

wait