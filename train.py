import numpy as np
import scipy.io as sio
from tdl import *
import random
import tensorly as tl
import pickle
random.seed(10)

#### Parameters ####
RR = np.array([0.1,0.1,1,1])              # Multilinear Rank of the tensor data as a percentage of the original dimensions - Size of the feature space
patch = np.array([3,3]).astype('int32')   # Patch size
n_sim = 5                                 # Number of simulations
bands = np.array([1,2,3,4,5,6,7,8,11,12]) # Highest resolution spectral bands
path_data = r".\Data\training_data.mat"   # Path to the training data

#### Training Data ####
img = sio.loadmat(path_data)
data = img['data']
data = data[:,:,bands,:]
# Take patches of the images
PM,num_patches = patches(data,patch)
data_tr = np.array(PM) # num_patches x patch[0] x patch[1] x num_bands x num_train_samples
# Concatenate 0 and 4 dimensions -> Total patches = num_patches * num_train_samples
data_tr = np.transpose(data_tr,(1,2,3,0,4))
data_tr = np.reshape(data_tr,(patch[0],patch[1],len(bands),-1))

#### Training Process ####
basis_mat = []     # Learned basis matrices
inv_basis_mat = [] # Inverse basis matrices
error = []         # Reconstruction error of the training data
# Perform n_sim simulations
for i in range(n_sim):
    print('Simulation {}/{}'.format(i+1,n_sim))
    # Perform Tensor Decomposition Learning
    PDnew, PDD, G_train, Xrec, er = Tensor_Decomposition_Learning(data_tr,RR,patch)
    basis_mat.append(PDnew)
    inv_basis_mat.append(PDD)
    error.append(er)

#### Save the learned basis matrices ####
with open(r'.\Data\learned_basis_matrices.pickle','wb') as fp:
    tr_models = {'basis_mat':basis_mat,'inv_basis_mat':inv_basis_mat}
    pickle.dump(tr_models,fp)
