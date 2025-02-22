import torch as tn

def QR(mat):
    """
    Compute the QR decomposition. Backend can be changed.

    Parameters
    ----------
    mat : tn array
        DESCRIPTION.

    Returns
    -------
    Q : the Q matrix
    R : the R matrix

    """
    Q,R = tn.linalg.qr(mat)
    return Q, R

def lr_orthogonal(tt_cores, R, instr):
    """Make cores left orthogonal in equivariant sence
        tt_cores = [torch.randn(num_paths, rank[r], Nc[r], rank[r+1]) for r in range(d)] 
        R - old ranks
        instr - list of instr
    """
    
    d = len(tt_cores)

    rank_next = R[0]
    
    core_now = tt_cores[0]
    cores_new = d*[None]
    
    
    # Loop over cores
    for i in range(d-1):
        # Init instr
        lmax = max([el[2] for el in instr[i]])
        ind_left = [[ii for ii, ir in enumerate(instr[i]) if ir[-1] == ll] for ll in range(lmax+1)]
        ind_right = [[ii for ii, ir in enumerate(instr[i+1]) if ir[0] == ll] for ll in range(lmax+1)]
        
        
        # Loop over orthogonalized momentum
        
        core_next = tt_cores[i+1]
        shape_next = list(core_next.shape[2:]) 
        for l in range(lmax+1):
            #print(core_now.shape)
            mode_shape = [core_now.shape[2]]
            
            core_now_tmp = tn.reshape(core_now[ind_left[l]],[len(ind_left[l])*core_now.shape[1]*core_now.shape[2],-1])
            
            #print(core_now_tmp.shape)
            # perform QR
            Qmat, Rmat = QR(core_now_tmp)
            core_now_tmp = Qmat
            
            
            #print(Qmat.shape)
            #print(Rmat.shape)
            # take next core 
            #print(len(ind_left[l]), len(ind_right[l]))
            
            core_next_tmp = tn.stack([core_next[ind, ...] for ind in ind_right[l]], dim = -3).flatten(1)
            core_next_tmp = Rmat @ core_next_tmp
            

            
            core_next_tmp = tn.reshape(core_next_tmp,[core_now_tmp.shape[1]] + [len(ind_right[l])] + shape_next)
            
            # update the cores
            if l == 0:
                cores_new[i] = tn.zeros([len(instr[i])] + [R[i]] + mode_shape + [core_now_tmp.shape[1]])
                cores_new[i + 1] = tn.zeros([len(instr[i+1])] + [core_now_tmp.shape[1]] + shape_next)


            print(cores_new[i].shape)
            cores_new[i][ind_left[l]] = tn.reshape(core_now_tmp,[len(ind_left[l])] + [R[i]]+mode_shape+[-1])
            R[i+1] = core_now_tmp.shape[1]
            # TODO: mb transpose works too
            #cores_new[i+1][ind_right[l]] = core_next_tmp.transpose(0, 1)
            cores_new[i+1][ind_right[l]] = tn.stack([core_next_tmp[:, i] for i in range(len(ind_right[l]))], dim = 0)

        core_now = core_next
      
    cores_new = [el for el in cores_new]
    return cores_new, R
