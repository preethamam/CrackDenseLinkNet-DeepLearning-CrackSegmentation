MODEL=$1
BACKBONE=$2
BATCH_SIZE=$3
EPOCHS=$4
DATA_DIR=$5
DEVICE_NUM=$6


echo $MODEL
echo $BACKBONE
echo $BATCH_SIZE
echo $EPOCHS
echo $DATA_DIR
echo $DEVICE_NUM

python train.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --data_dir $DATA_DIR \
    --device_num $DEVICE_NUM \


