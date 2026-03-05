# Unsupervised-Change-Detection-On-Multi-Temporal-Satellite-Images-Using-Tensor-Decomposition-Learning

This repository contains Python codes and scripts designed for the unsupervised change detection of extreme events in multi-temporal satellite images based on a tensor decomposition learning method that uses the Tucker decomposition, as presented in the paper ["Unsupervised Change Detection on Multi-temporal Satellite Images Using Tensor Decomposition Learning" (A. Aidini, G. Tsagkatakis, P. Tsakalides)](https://users.ics.forth.gr/tsakalid/PAPERS/CNFRS/2024-IGARSS-Aidini.pdf). In the proposed method, we learn a basis matrix for each dimension of the feature space of the images using tensor decomposition learning. Then, each new image is represented in the feature space by expressing it as a multilinear combination of the learned tensor decomposition factors. The predicted changes can be obtained by comparing and thresholding the distance of the corresponding extracted features of the images before and after the event. 

## Requirements
### Datasets
To evaluate the proposed unsupervised change detection method, we initially trained our tensor decomposition learning model using Sentinel-2 multispectral images provided in the `training_data.mat` file of the `Data` folder of this repository. 
During inference, we performed experiments on the [RaVÆn dataset](https://drive.google.com/drive/folders/1VEf49IDYFXGKcfvMsfh33VSiyx5MpHEn). Specifically, we performed experiments on 5 different locations of fires and 4 locations of floods. Every location comprises a time series containing five images. The initial four images are captured before the disaster occurs, while the fifth image is taken afterward.

### Framework
We use the library [TensorLy](http://tensorly.org/): Tensor Learning in Python to execute tensor operations.

## Contents
`train.py`: Perform the training process to learn the basis matrices that synthesize the feature space from available 
           satellite data using the tensor decomposition learning method.

`test.py`: Perform the inference phase to predict the changes of extreme events in multi-temporal satellite images. 
           Specifically, we represent each image in the feature space by expressing it as a multilinear combination of 
           the acquired basis matrices, and we compare and threshold the distance of the extracted features from the 
           images before and after the event.

`tdl.py`: All necessary functions related to the tensor decomposition learning method. 

`data.py`: All necessary functions to form the time series and the corresponding mask of the true changes of a 
           location of an event. 

## Execution
### Training Process: 
We run the `train.py` script, which takes the multilinear rank, the patch size, the number of simulations, the selected spectral bands, and the path to the training data, and saves the learned basis matrices in the file `learned_basis_matrices.pickle` in the `Data` folder.

### Run-time Phase:
We run the `test.py` script which takes the multilinear rank, the patch size, the number of simulations, the number of history frames, the selected spectral bands, the path to the learned basis matrices and the path to the data of an event (e.g. floods) of the RaVÆn dataset, and computes the mean Area Under Precision-Recall Curve (AUPRC) and the corresponding standard deviation of the proposed method.

### References
If you use our method, please cite:
```
@inproceedings{aidini2024unsupervised,
  title={Unsupervised Change Detection on Multi-Temporal Satellite Images Using Tensor Decomposition Learning},
  author={Aidini, Anastasia and Tsagkatakis, Grigorios and Tsakalides, Panagiotis},
  booktitle={IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium},
  pages={8495--8499},
  year={2024},
  organization={IEEE}
}
```
If you use the dataset, please cite:
```
@article{ruuvzivcka2022ravaen,
  title={RaV{\AE}n: unsupervised change detection of extreme events using ML on-board satellites},
  author={R{\uu}{\v{z}}i{\v{c}}ka, V{\'\i}t and Vaughan, Anna and De Martini, Daniele and Fulton, James and Salvatelli, Valentina and Bridges, Chris and Mateo-Garcia, Gonzalo and Zantedeschi, Valentina},
  journal={Scientific reports},
  volume={12},
  number={1},
  pages={16939},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```
