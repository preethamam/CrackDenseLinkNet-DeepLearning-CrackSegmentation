DATA=$1
DEVICE_NUM=$2
DATA_DIR="/media/preethamam/Utilities-SSD-1/Xtreme_Programming/ZZZ_Data/DLCrack/Liu+Xincong+DS3+CrackSegNet/Testing/"
ALL="all"

echo $DATA
echo $DEVICE_NUM

if [ $DATA = "all" ]; then

    echo 1
    python test.py \
        --model Linknet\
        --backbone densenet169 \
        --data_dir "${DATA_DIR}Liu" \
        --device_num $DEVICE_NUM

    python performance_analysis.py \
        --data "${DATA_DIR}Liu"

    python test.py \
        --model Linknet\
        --backbone densenet169 \
        --data_dir "${DATA_DIR}CrackSegNet" \
        --device_num $DEVICE_NUM
    
    python performance_analysis.py \
        --data "${DATA_DIR}CrackSegNet"
    
    python test.py \
        --model Linknet\
        --backbone densenet169 \
        --data_dir "${DATA_DIR}DS3" \
        --device_num $DEVICE_NUM

    python performance_analysis.py \
        --data "DS3"

    python test.py \
        --model Linknet\
        --backbone densenet169 \
        --data_dir "${DATA_DIR}Xincong" \
        --device_num $DEVICE_NUM
    
    python performance_analysis.py \
        --data "${DATA_DIR}Xincong"

else        
    # python test.py \
    #     --model Linknet\
    #     --backbone densenet169 \
    #     --data_dir "${DATA_DIR}${DATA}" \
    #     --device_num $DEVICE_NUM
    
    python test.py \
        --model Linknet\
        --backbone densenet169 \
        --data_dir "${DATA_DIR}${DATA}" \
        --device_num $DEVICE_NUM

    python performance_analysis.py \
        --data "${DATA_DIR}${DATA}"
fi
