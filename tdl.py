import numpy as np
import tensorly as tl
import math

def patches(M,patch_size):
    # Divide a tensor into patches
    # Input:
    #       M          : I_1 x I_2 x ... x I_N original tensor
    #       patch_size : 2 x 1 vector of the spatial size of each patch
    # Output:
    #       PM         : list with the patches of the tensor

    dim = M.shape
    N = M.ndim
    
    n1 = math.ceil(dim[0]/patch_size[0])
    n2 = math.ceil(dim[1]/patch_size[1])

    num_patches = n1*n2
    PM = []
    for i in range(n1):
        x1 = i*patch_size[0]
        if i == (n1-1):
            x2 = dim[0]
        else:
            x2 = (i+1)*patch_size[0]
        for j in range(n2):
            y1 = j*patch_size[1]
            if j == (n2-1):
                y2 = dim[1]
            else:
                y2 = (j+1)*patch_size[1]
            if N == 2:
                PM.append(M[x1:x2,y1:y2])
            elif N == 3:
                PM.append(M[x1:x2,y1:y2,:])
            elif N == 4:
                PM.append(M[x1:x2,y1:y2,:,:])
            elif N == 5:
                PM.append(M[x1:x2,y1:y2,:,:,:])
            else:
                raise TypeError('The tensor are not 2D, or 3D or 4D or 5D')
    return PM,num_patches

def union_patches(PM,dim,patch_size):
    # Unite the patches to form a tensor 
    # Input:
    #       PM         : list with the patches of the tensor
    #       dim        : dimensions of the original tensor (vector)
    #       patch_size : 2 x 1 vector of the spatial size of each patch
    # Output:
    #       M          : unified tensor

    N = len(dim)

    n1 = math.ceil(dim[0]/patch_size[0])
    n2 = math.ceil(dim[1]/patch_size[1])

    M = np.zeros(dim)
    k = 0
    for i in range(n1):
        x1 = i*patch_size[0]
        if i == (n1-1):
            x2 = dim[0]
        else:
            x2 = (i+1)*patch_size[0]
        for j in range(n2):
            y1 = j*patch_size[1]
            if j == (n2-1):
                y2 = dim[1]
            else:
                y2 = (j+1)*patch_size[1]
            if N == 2:
                M[x1:x2,y1:y2] = PM[k]
            elif N == 3:
                M[x1:x2,y1:y2,:] = PM[k]
            elif N == 4:
                M[x1:x2,y1:y2,:,:] = PM[k]
            elif N == 5:
                M[x1:x2,y1:y2,:,:,:] = PM[k]
            else:
                raise TypeError('The tensor are not 2D, or 3D or 4D or 5D.')
            k += 1
    return M

def size_core(dim,patch_size,RR):
    # Estimate the size of the core tensor
    # Inputs:
    #        dim        : dimensions of the original data (vector)
    #        patch_size : 2 x 1 vector of the spatial size of each patch
    #        RR         : dimensions of the core tensor as
    #                     percentage of the original dimensions (vector)
    #
    # Output:
    #        size_G : size of the core tensor (vector)

    # number of full-sized patches in the spatial dimensions x, y
    fullpatch_x = math.floor(dim[0]/patch_size[0])
    fullpatch_y = math.floor(dim[1]/patch_size[1])
    # number of no-full-sized patches in the spatial dimensions x, y
    smallpatch_x = dim[0]-fullpatch_x*patch_size[0]
    smallpatch_y = dim[1]-fullpatch_y*patch_size[1]

    # size of the whole core tensor
    size_G = np.ceil(RR[:len(dim)]*dim).astype('uint16')
    size_G[0] = fullpatch_x*np.ceil(RR[0]*patch_size[0])+np.ceil(RR[0]*smallpatch_x)
    size_G[1] = fullpatch_y*np.ceil(RR[1]*patch_size[1])+np.ceil(RR[1]*smallpatch_y)

    return size_G.astype('int32')

def Tensor_Decomposition_Learning(Xtrain,RR,patch_size):
    # Learn the basis matrices of each dimension from training samples
    # Inputs:
    #        Xtrain     : (N+1)-way tensor that contains the N-way training tensors
    #                     (the last dimension indicates the samples)
    #        RR         : (N+1) x 1 vector of the dimensions of the core tensor as
    #                     percentage of the original dimensions
    #        patch_size : 2 x 1 vector of the spatial size of each patch
    #
    # Outputs:
    #        PDnew : Learned basis matrices (N x 1 list)
    #        PDD   : Inverted learned basis matrices (N x 1 list)
    #        Grec  : (N+1)-way tensor that contains the core tensors of the samples
    #        Xrec  : (N+1)-way tensor that contains the reconstructed training tensors
    #        er    : num_patches x 1 list of the NRMSE of the patches at
    #                each iteration

    N = Xtrain.ndim-1
    dim = Xtrain.shape

    # Parameters
    itter = 50 # number of maximum iterations
    p = 0.01   # step-size parameter (0.01)
    tol = 1e-7 # tolerance for stopping criterion

    # Training Process - ADMM - min L(G,D{1},..,D{N},A{1},..,A{N},Y{1},..,Y{N})
    # iteratively, with respect to each variable
    print('Learn the basis matrices of each dimension from training samples')

    # Divide Xtrain into patches
    PXtrain, num_patches = patches(Xtrain,patch_size)

    PDnew = []
    PDD = []
    PMrec = []
    er = []
    Gnew = []
    r = np.zeros(num_patches).astype('uint16')
    for k in range(num_patches):
        PDnew.append([])
        PDD.append([])
        # print('Patch {}/{}'.format(k+1,num_patches))
        pdim = PXtrain[k].shape
        R = np.ceil(RR*pdim).astype('uint16')
        # Initialization of the variables
        D = []
        DD = []
        for n in range(N+1):
            if R[n] == 0:
                R[n] = 1
            D.append(np.random.rand(pdim[n],R[n]))
            if n!= N:
                Q, rq = np.linalg.qr(D[n])
                D[n] = Q
                DD.append(D[n].T)
            else:
                DD.append(np.linalg.pinv(D[N]))
        A = D[N]
        G = tl.tenalg.multi_mode_dot(PXtrain[k],DD)
        Y = np.zeros(D[N].shape)
        for n in range(N):
            PDnew[k].append(D[n])
            PDD[k].append(DD[n])
        PMrec.append(tl.tenalg.multi_mode_dot(G,D))

        er.append([])
        for i in range(itter):
            # print('Iteration: {}'.format(i+1))
            # Update A
            A = D[N]-(1/p)*Y
            U1,s1,V1 = np.linalg.svd(A,full_matrices=True)
            ss = np.cumsum(s1)/sum(s1)
            rr = len(np.where(ss<0.90)[0])
            if i == 0:
                r[k] = rr
            if rr == 0:
                r[k] = 1
            elif rr < r[k]:
                r[k] = rr
            S1 = np.diag(s1)
            A = U1[:,:r[k]]@S1[:r[k],:r[k]]@V1[:,:r[k]].T
                
            # Update D{1},..,D{N},D{N+1}
            Dnew = []
            for n in range(N+1):
                CN = tl.tenalg.multi_mode_dot(G,D,skip=n)
                Cnn = tl.unfold(CN,n)
                if n == N:
                    Dnew.append((tl.unfold(PXtrain[k],n)@Cnn.T+Y+p*A)@np.linalg.pinv(Cnn@Cnn.T+p*np.eye(R[n])))
                else:
                    Dnew.append((tl.unfold(PXtrain[k],n)@Cnn.T)@np.linalg.pinv(Cnn@Cnn.T))
                    Q, rq = np.linalg.qr(Dnew[n])
                    Dnew[n] = Q
            D = Dnew
            # Update G
            for n in range(N):
                DD[n] = D[n].T
            DD[N] = np.linalg.pinv(D[N])
            G = tl.tenalg.multi_mode_dot(PXtrain[k],DD)
            # Update Y
            Y = Y+p*(A-D[N])

            # Stopping criterion
            PMrec[k] = tl.tenalg.multi_mode_dot(G,D)
            if np.array_equal(PMrec[k],PXtrain[k]):
                er[k].append(0)
            else:
                er[k].append(np.linalg.norm(PMrec[k].flatten()-PXtrain[k].flatten())/np.linalg.norm(PXtrain[k].flatten()))
            if (i>0)and(er[k][i]>er[k][i-1]):
                break
            else:
                for n in range(N):
                    PDnew[k][n] = D[n]
                    PDD[k][n] = DD[n]
                if (er[k][i] <= tol)or((i>0)and(abs(er[k][i]-er[k][i-1])<=1e-4)):
                    break
        Gnew.append(tl.tenalg.multi_mode_dot(PXtrain[k],PDD[k],skip=N))
    # Estimate the reconstructed training tensor
    Xrec = union_patches(PMrec,Xtrain.shape,patch_size)
    size_G = size_core(dim,patch_size,RR)
    Grec = union_patches(Gnew,size_G,np.ceil(RR[:1]*patch_size).astype('int32'))
    err = np.linalg.norm(Xrec.flatten()-Xtrain.flatten())/np.linalg.norm(Xtrain.flatten())
    print('Training NRMSE of the Tensor Decomposition Learning process: {}'.format(err))
    
    return PDnew, PDD, Grec, Xrec, er

def Estimate_core(Xtest,PDnew,PDD,patch_size,RR):
    # Estimate the core tensor of the samples using the learned basis matrices
    # Inputs:
    #        Xtest      : (N+1)-way tensor that contains the N-way testing tensors
    #                     (the last dimension indicates the samples)
    #        PDnew      : Learned basis matrices (N x 1 list)
    #        PDD        : Inverted learned basis matrices (N x 1 list)
    #        patch_size : 2 x 1 vector of the spatial size of each patch
    #        RR         : (N+1) x 1 vector of the dimensions of the core tensor as
    #                     percentage of the original dimensions
    #
    # Outputs:
    #        G    : num_test x 1 list of num_patches x 1 lists of the core tensors of the testing patches
    #        Mrec : num_test x 1 list of the reconstructed testing samples
    #        nmse : num_test x 1 vector of the NRMSE of the reconstructed testing samples

    # print('Estimate the core tensor of the testing samples using the learned basis matrices')

    dim = Xtest.shape
    if len(dim)<len(RR):
        N = Xtest.ndim
        num_test = 1
    else:
        N = Xtest.ndim-1
        dim = dim[:N]
        num_test = Xtest.shape[N]


    nmse = np.zeros(num_test)
    Mrec = []
    G = []
    if N == 1:
        size_G = np.ceil(RR*Xtest.shape).astype('uint16')
        Grec = np.zeros(size_G.shape)
        for j in range(num_test):
            test = Xtest[:,j]
            G.append(PDD[0][0]@test)
            Grec[:,j] = G[j]
            Mrec.append(PDnew[0][0]@Grec[:,j])
            nmse[j] = np.linalg.norm(Mrec[j].flatten()-test.flatten())/np.linalg.norm(test.flatten())
    else:
        if len(Xtest.shape)==N:
            Xtest = np.expand_dims(Xtest,axis=3)

        size_G = size_core(Xtest.shape,patch_size,RR)

        Xtest = tl.unfold(Xtest,N)
        for j in range(num_test):
            # print('Testing Sample {}'.format(j))
            # Take the j-th testing sample
            test = Xtest[j,:]
            test = np.reshape(test,dim)
            # Divide the testing sample into patches
            PMtest,num_patches = patches(test,patch_size)

            PMrec = []
            GG = []
            for k in range(num_patches):
                # Estimate the core tensor
                GG.append(tl.tenalg.multi_mode_dot(PMtest[k],PDD[k]))
                # Compute the reconstructed testing sample
                if GG[k].ndim<N:
                    mm = np.zeros(N)
                    for n in range(N):
                        mm[n] = PDnew[k][n].shape[1]
                    PMrec.append(GG[k])
                    for n in range(N):
                        g = PDnew[k][n]@tl.unfold(PMrec[k],n)
                        mm[n] = PDnew[k][n].shape[0]
                        PMrec[k] = tl.fold(g,n,mm)
                else:
                    PMrec.append(tl.tenalg.multi_mode_dot(GG[k],PDnew[k]))
            G.append(GG)
            Mrec.append(union_patches(PMrec,test.shape,patch_size))
            if np.array_equal(Mrec[j],test):
                nmse[j] = 0
            else:
                nmse[j] = np.linalg.norm(Mrec[j].flatten()-test.flatten())/np.linalg.norm(test.flatten())
    print('Mean testing NRMSE from Tensor Decomposition Learning {}'.format(np.mean(nmse)))
    
    return G, Mrec, nmse