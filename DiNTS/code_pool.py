import numpy as np
import pdb
from copy import deepcopy
def gen_mtx(depth=3):
    # total path in a block
    paths = 3*depth-2
    # use depth first search to find all path activation combination
    def dfs(node, paths=6):
        if node == paths:
            return [[0],[1]]
        else:
            child = dfs(node+1, paths)
            return [[0] + _ for _ in child]+ [[1] + _ for _ in child]
    all_connect = dfs(0, paths-1)
    # Save all possible connections in mtx (might be redundant and infeasible)
    mtx = []
    for _ in all_connect:
        # convert path activation [1,paths] to path activation matrix [depth, depth]
        ma = np.zeros((depth,depth))
        for i in range(paths):
            ma[(i+1)//3, (i+1)//3-1 + (i+1)%3] = _[i]
        mtx.append(ma)    
    
    # Calculate path activation to node activation params
    tidx, code2in, code2out =  [], [], []
    for i in range(paths):
        tidx.append((i+1)//3 * depth + (i+1)//3-1 + (i+1)%3) 
        code2in.append((i+1)//3-1 + (i+1)%3)
    code2ops = ([-1,0,1]*depth)[1:-1]
    for _ in range(depth):    
        code2out.extend([_,_,_])
    code2out = code2out[1:-1]
    
    # define all possible node activativation
    node_act_list = dfs(0,depth-1)[1:]
    transfer_mtx = {}
    for code in node_act_list:
        # make sure each activated node has an active connection, inactivated node has no connection
        code_mtx = [_ for _ in mtx if ((np.sum(_,0)>0).astype(int)==np.array(code)).all()]
        transfer_mtx[str(np.array(code))] = code_mtx

    return transfer_mtx, node_act_list, tidx, code2in, code2ops, code2out, all_connect[1:]
