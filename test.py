import numpy as np
import scipy.io as sio
import os
from tdl import *
from data import *
import pickle
from pysptools import distance
from sklearn.metrics import precision_recall_curve, auc

# Parameters
RR = np.array([0.1,0.1,1,1])                         # Multilinear Rank of the tensor data as a percentage of the original dimensions - Size of the feature space
patch = np.array([3,3]).astype('int32')              # Patch size
n_sim = 5                                            # Number of simulations
history = 3                                          # Number of history frames
bands = np.array([1,2,3,4,5,6,7,8,11,12])            # Highest resolution spectral bands
path_model = r'.\Data\learned_basis_matrices.pickle' # Path to the learned basis matrices
path_data = r'.\Data\floods'                         # Path to the data of an event

# Load Training Models
with open(path_model,'rb') as fp:
    tr_models = pickle.load(fp)
basis_mat = tr_models['basis_mat'] # Basis matrices
inv_basis_mat = tr_models['inv_basis_mat'] # Inverse basis matrices

# Initiallization
time_series = os.listdir(path_data)                    # Different locations of the event
data_te_all = []                                       # Time series of the selected frames
changes = []                                           # True changes
auprc = np.zeros((len(time_series),n_sim))             # AUPRC of the predictions 
rec_err = np.zeros((len(time_series),n_sim,history+1)) # Reconstruction error of the expression of the images in the feature space
pred_scores = []                                       # Predicted scores
for t in range(len(time_series)): # for each location of an event
    # Form the times series and the corresponding mask of true changes
    img = form_time_series(os.path.join(path_data,time_series[t]),bands)
    data_te = img['data'][:,:,:,:4] # Take the frames before the event

    # Select the frames with the lowest cloud cover (the frames with the most small values compared to the other frames)
    ii = np.argmax(data_te[:,:,0,:],axis=2)
    inv = np.where(img['changes']==2)
    ii[inv] = -1
    l = np.zeros((data_te.shape[3]))
    for i in range(data_te.shape[3]):
        l[i] = len(np.where(ii==i)[0])
    frames = np.argsort(l)[:history]
    data_te = data_te[:,:,:,frames]
    print('Selected Frames: {}'.format(frames))
    
    # Concatenate the selected frames with the frame after the event
    data_te = np.concatenate((data_te,np.expand_dims(img['data'][:,:,:,4],axis=3)),axis=3)
    if len(data_te.shape)<4:
        data_te = np.expand_dims(data_te,axis=3)
    
    # Take the images and corresponding mask of true changes by dropping the last patch
    data_te_all.append(data_te[:(data_te.shape[0]-data_te.shape[0]%patch[0]),:(data_te.shape[1]-data_te.shape[1]%patch[1]),:,:])
    changes.append(img['changes'][:data_te_all[t].shape[0],:data_te_all[t].shape[1]])

    # Predict changes and calculate AUPRC in n_sim simulations
    pred_scores.append([])
    for iter in range(n_sim):
        G = [] # Coefficients of each patch of each frame
        for i in range(data_te_all[t].shape[3]): # for each frame
            # Get the patches of the data
            PM,num_patches = patches(data_te_all[t][:,:,:,i],patch)
            data = np.transpose(np.array(PM),(1,2,3,0))

            # Each patch of the image is represented in the feature space by expressing it as a multilinear combination of the learned basis matrices
            # Estimate the corresponding coefficients and the reconstruction of the testing sample 
            G_test, PMrec, nmse = Estimate_core(data,basis_mat[iter],inv_basis_mat[iter],patch,RR)
            G.append(G_test)
            rec_err[t,iter,i] = np.mean(nmse)

        # Find the predicted change scores between the last image and the previous images in the time series
        pred_change = np.zeros((data_te_all[t].shape[0],data_te_all[t].shape[1],history)) # Predicted scores for each history frame         
        for i in range(history): # for each history frame
            pred = []
            for k in range(len(G[0])): # for each patch of the image
                pred.append(np.ones(patch))
                # Compute the distance of the corresponding coefficients of the images before and after the event.
                p = np.zeros((G[0][k][0].shape[0],G[0][k][0].shape[1]))
                for x in range(G[0][k][0].shape[0]):
                    for y in range(G[0][k][0].shape[1]):
                        if np.array_equal(np.squeeze(G[i][k][0][x,y,:]),np.squeeze(G[data_te_all[t].shape[3]-1][k][0][x,y,:]))or(np.unique(G[data_te_all[t].shape[3]-1][k][0][x,y,:]).any()==0)or(np.unique(G[i][k][0][x,y,:]).any()==0):
                            p[x,y] = 0
                        else:
                            p[x,y] = distance.SAM(np.squeeze(G[i][k][0][x,y,:]),np.squeeze(G[data_te_all[t].shape[3]-1][k][0][x,y,:]))
                pred[k] = pred[k]*np.mean(p)
            # Unite the patches to form the image with the predicted scores
            pred_change[:,:,i] = union_patches(pred,data_te_all[t].shape[:2],patch)

        # Take the minimum of the scores  
        pred_change_scores = np.min(pred_change,axis=2)
        pred_scores[t].append(pred_change_scores)
        # Compute the AUPRC
        invalid_masks  = [c==2 for c in changes[t]]
        true_changes = np.concatenate([c[~m] for m,c in zip(invalid_masks, changes[t])],axis=0)
        pred_change_scores = np.concatenate([c[~m] for m,c in zip(invalid_masks, pred_change_scores)],axis=0)
        precision, recall, thresholds = precision_recall_curve(true_changes,pred_change_scores)
        auprc[t,iter] = auc(recall, precision)
        print('Time series {}, Iteration {}: AUPRC {}'.format(t+1,iter+1,auprc[t,iter]))

mo = np.mean(auprc,axis=0)
print('Mean AUPRC {} and standard deviation {}'.format(np.mean(mo),np.std(mo)))
