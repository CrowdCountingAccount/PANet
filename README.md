# Main Codes for PANet
The main logic of this project is shown in *main.py*. Please read this file before using this project.
## Datasets
Choose the dataset using `--dataset`. There are four datasets, including `SHHA` for ShanghaiTech Part A, `SHHA` for ShanghaiTech Part B, `QNRF` for UCF-QNRF and `CC50` for UCF-CC-50.

Make the data directories according to *main.py*.

The Gaussian density maps are in *.npy* format. For the SHHA dataset, we use adaptive sigma. (Refer to the paper.)

You need to divide the UCF-CC-50 dataset into the training set and the test set according to the 5-fold validation strategy.

You need to resize the images and density maps in the QNRF dataset according to *resize.py*.

## Training

There are three training stages, distinguished by `--stage`, including `rough`, `teacher` and `student`.

`rough` is for training the rough network, supervised by rough density maps (sigma=50). After this stage, the rough density predictions are saved.

`teacher` is for training the precise network (PANet), supervised by precise density maps. This is the first stage of SDS. The rough density predictions are loaded to generate the dilation maps. After this stage, the predictions after count correction are saved as the supervision targets for the next stage.

`student` is for training the precise network (PANet), supervised by the new ground truth generated in the previous stage. This is the second stage of SDS. The rough density predictions are loaded to generate the dilation maps. In this stage, the precise network outputs the final predictions.

## Inference

There are two stages for inference. We generate the rough density predictions on the test set in *main.py*, and save them in the corresponding directories. So in *inference.py*, there is only the second stage, i.e., loading the rough density predictions and outputs the predictions by the precise network. 

For example, you can test PANet by the following command:
```
python inference.py --cuda=0 --dataset=QNRF --checkpoint=your_checkpoint_dir/model.pt --dilation-path=your_dilation_path/QNRF/test/rough/ --data-dir=./data/
```

Actually we output the predictions on the test set in every training epoch.