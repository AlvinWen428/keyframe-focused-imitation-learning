Keyframe-Focused Visual Imitation Learning
-------------------------------------------------------------

This is the code of the paper 
[Keyframe-Focused Visual Imitation Learning](http://proceedings.mlr.press/v139/wen21d/wen21d.pdf). 
You can use this repo to reproduce the results of BC-SO (behavioral cloning with single observation),
BC-OH (behavioral cloning with observation history) and our method.


### Getting started

#### Installation

To install the required python packages, just run:

    conda create -n carla python=3.6
    conda activate carla
    bash setup.sh

#### Setting Environment/ Getting Data

The first thing you need to do is define the datasets folder.
This is the folder that will contain your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>

Download a sample dataset pack, with one training
and two validations, by running

    python3 tools/get_sample_datasets.py

The datasets: CoILTrain, CoILVal1 and CoILVal2; will be stored at
 the COIL_DATASET_PATH folder.
 
The dataset used in our paper is CARLA100, which can be downloaded from 
the original [github repo](https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md) 
of Felipe Codevilla. Download the .zip files and extract the dataset into $COIL_DATASET_PATH/CoilTrain100.

To collect other datasets please check the data collector repository.
https://github.com/carla-simulator/data-collector


#### Executing

You are suggested to run BC-OH baseline first 
because the preload .npy file for the dataset will be generated in this process 
(we have provided it in the _preload/ ).

##### Run BC-OH baseline:
    
    # train
    python3 coiltraine.py --folder wen_bcoh_nospeed -e retrain1 --single-process train -de NocrashTrainingDenseRep3_Town01 -vd CoILVal1 --gpus 0 --docker carlasim/carla:0.8.4
    # evaluate
    python3 coiltraine.py --folder wen_bcoh_nospeed -e retrain1 --single-process drive -de NocrashTrainingDenseRep3_Town01 -vd CoILVal1 --gpus 0 --docker carlasim/carla:0.8.4


##### Get APE score as importance weight:

You can run ``python3 action_correlation/get_importance_weights.py [args]`` to generate the APE score for each sample 
(we have provided one in the _preload/ ).

See ``python3 action_correlation/get_importance_weights.py -h`` for the detail of the arguments.

And then you can run our method.

##### Run our method:

    # train
    python3 coiltraine.py --folder ours_threshold_weight5_nospeed -e resnet_7frame_retrain1 --single-process train -de NocrashTrainingDenseRep3_Town01 -vd CoILVal1 --gpus 0 --docker carlasim/carla:0.8.4
    # evaluate
    python3 coiltraine.py --folder ours_threshold_weight5_nospeed -e resnet_7frame_retrain1 --single-process drive -de NocrashTrainingDenseRep3_Town01 -vd CoILVal1 --gpus 0 --docker carlasim/carla:0.8.4

We have uploaded the example checkpoints of BCOH and our model [here](https://drive.google.com/drive/folders/17xLzi8-AQm2cz0gsmc1gLIWxrLFRQFeO?usp=sharing), 
you can download them into ``_logs/`` folder and then directly evaluate them.

For the evaluation benchmark, we retest all the methods for three times in NoCrashDense benchmark, 
i.e. ``-de NocrashTrainingDenseRep3_Town01``, because of the high variance of CARLA simulator.
After the evaluation is over, you can use ``tools/print_metrics.py`` to get the statistical metrics.

For more details about the arguments and how to execute the experiments, please refer to the README in the original carla github [repo](https://github.com/felipecode/coiltraine).

## Citations
Please consider citing our paper in your publications if it helps. Here is the bibtex:

```
@InProceedings{wen2021keframe,
  title = 	 {Keyframe-Focused Visual Imitation Learning},
  author =       {Wen, Chuan and Lin, Jierui and Qian, Jianing and Gao, Yang and Jayaraman, Dinesh},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {11123--11133},
  year = 	 {2021},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR}
}
```

