# Setting up Environment

Please make sure you have installed Anaconda and have Python version `3.8.5` before following next steps

```shell
cd conda_venv
conda env create -f environment.yml
conda activate dl_crack

# Install dependencies
cat requirements.txt | xargs -n 1 pip install

# Work on the project

# Deactivate the virtual environment when you are done
conda deactivate

```

Note that few of the path specific packages may fail during `pip`. Ignore those failure messages and let it terminate by itself. It should install all the necessary packages required to run the below documented scripts.

However, by any chance if still you get package errors then just install that package through pip or conda.

# Training Script

Below shell command will invoke the `tail_train.py` i.e. model training code as a backgroud process. You need to change the variables in the output log file name in given script

```console
./train.sh Linknet densenet169 8 2 './data/' 0 > ./output_log/[MODEL_NAME]_[BACKBONE_NAME]_b[BATCH_SIZE]e[NUM_EPOCHS] 2>&1 &
```

Model weights are stored in `logs` dir in respective folder based on model and backbone arguments.

Verify the path to data directory before invoking training script. Assuming you linked the data directory `'./data'` following should be the hierarchy of folders with the exact folder names as mentioned below:

* data
    * TrainingCracks
    * TrainingCracksGroundtruth
    * ValidationCracks
    * ValidationCracksGroundtruth
    * TestingCracks
    * TestingCracksGroundtruth


# Testing Script

Below shell command will invoke the `tail_test.py` i.e. model training code as a backgroud process

```console
./test.sh Linknet densenet169 './data/' 0
```

Verify the path to data directory before invoking training script. Assuming you linked the data directory `'./data'` following should be the hierarchy of folders with the exact folder names as mentioned below:

* data
    * TrainingCracks
    * TrainingCracksGroundtruth
    * ValidationCracks
    * ValidationCracksGroundtruth
    * TestingCracks
    * TestingCracksGroundtruth

The script will load the model based on the model arguments from the `logs` folder and run an inference on Testing Images from `./data/` directory. Post that, it will save the segmentation results and ground truth in the `pred` folder in the following manner:

* pred
    * tail
        * seg   
        * gnd

`seg` containes inference (segmented) images and gnd contains the ground truth images which were already given. It stores ground truth again in this hierarachy for the sake of simplicity while running performance script as explained below:



# Performance Script

For performance analysis, you simply need to run the following command:

```console
python performance_analysis.py > ./performance_logs/[ModelName]_[BackboneName].txt
```

Make sure to change the `ModelName` and `BackboneName` to the respective model and backbone names for which you invoke this script.

`performance_analysis.py` just needs `pred` folder populated with segmented images and ground truth images in the mentioned hierarchy above. Hence, you do not need so specify any arguments while calling this script.

Once it has computed the metrics, it should output the metrics in the txt file with the given output name (through shell command) in the `performance_logs` folder.


# Map Extraction Script

Run the following command for invoking `map_extract.py` (change the arguments as per requirements)

```console
python map_extract.py --model Linknet --backbone densenet169 --device_num 0 --data_dir './data/ --image_name '11215-11.jpg'
```

It extracts feature maps and filter maps and stores it in the respective `./feature_maps` and `./filter_maps` folder.