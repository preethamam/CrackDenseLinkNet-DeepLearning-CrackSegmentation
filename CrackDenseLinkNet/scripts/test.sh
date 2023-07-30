MODEL=$1
BACKBONE=$2
DATA_DIR=$3
DEVICE_NUM=$4


echo $MODEL
echo $BACKBONE
echo $DATA_DIR
echo $DEVICE_NUM

python tail_test.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --data_dir $DATA_DIR \
    --device_num $DEVICE_NUM
