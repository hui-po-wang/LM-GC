cd ..

DATASET='tinyimagenet' # cifar10 # mnist
ARCH="convnet" # vgg16 # resnet18 # vit
for i in 0 1 2
do
    python -u train_and_collect_grad.py -cfg settings/gradient_collection/$DATASET-$ARCH.yaml --tag $i --grad-interval 400 --download
done