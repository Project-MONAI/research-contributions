#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from operations import Cell
from code_pool import gen_mtx
import copy
import random
import pdb


# class Interpolate(nn.Module):
#     def __init__(self, scale_factor, mode, align_corners):
#         super(Interpolate, self).__init__()

#         self.align_corners = align_corners
#         self.interp = F.interpolate
#         self.mode = mode
#         self.scale_factor = scale_factor
        
#     def forward(self, x):
#         x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
#         return x


class AutoUnet(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks = 6, num_depths=3, cell = Cell, cell_ops=5, k=1, \
                 channel_mul = 1.0, affine=True, use_unet=False, probs=0.9, ef=0.3, use_stem=False, code=None):
        """ Initialize NAS network search space
        Args:
            in_channels: input image channel
            num_classes: number of segmentation classes
            num_blocks: number of blocks (depth in the horizontal direction)
            num_depths: number of image resolutions: 1, 1/2, 1/4 ... in each dimention, each resolution feature is a node at each block
            cell: operatoin of each node
            cell_ops: cell operation numbers
            k: PC-Darts channel partial rate
            channel_mul: adjust intermediate channel number, default 1.
            affine: if true, use affine in instance norm. 
            use_unet/probs: initialize path/cell probabilities (log_alpha) with probs as a U-Net. 
            ef: early fix threshold. If ef >= 1, then not using early fix. 
            code: [node_a, code_a, code_c] decoded using self.decode(). Remove unused cells in retraining
        Predefined variables:        
            filter_nums: default init 64. Double channel number after downsample
            
            topology related varaibles from gen_mtx():
                trans_mtx: feasible path activation given node activation key
                code2in: path activation to its incoming node index
                code2ops: path activation to operations of upsample 1, keep 0, downsample -1
                code2out: path activation to its output node index
                node_act_list: all node activation codes [2^num_depths-1, res_num]
                node_act_dict: node activation code to its index 
                tidx: index used to convert path activation matrix (depth,depth) in trans_mtx to path activation code (1,3*depth-2)
        """
        super(AutoUnet, self).__init__()
        self.istrain = True
        # predefined variables
        filter_nums = [int(32 * channel_mul), int(64 * channel_mul), int(128 * channel_mul), int(256 * channel_mul), int(512 * channel_mul)]  
        # path activation and node activations
        trans_mtx, node_act_list, tidx, code2in, code2ops, code2out, child_list = gen_mtx(num_depths)
        node_act_list = np.array(node_act_list)  
        node_act_dict = {str(node_act_list[i]):i for i in range(len(node_act_list))}
        self.num_depths = num_depths
        self.filter_nums = filter_nums
        self.cell_ops = cell_ops
        self.code2in = code2in
        self.code2ops = code2ops
        self.code2out = code2out
        self.node_act_list = node_act_list
        self.node_act_dict = node_act_dict
        self.trans_mtx = trans_mtx
        self.tidx = tidx
        self.child_list = np.array(child_list)
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_depths = num_depths
        self.affine = affine
        self.ef = ef
        self.probs = probs
        self.use_unet = use_unet
        self.use_stem = use_stem

        # define stem operations for every block
        self.stem_down = nn.ModuleDict() 
        self.stem_up = nn.ModuleDict()
        self.stem_finals = nn.Sequential(nn.ReLU(),
                                         nn.Conv3d(filter_nums[0],filter_nums[0], 3, stride=1, padding=1, bias=False),
                                         nn.InstanceNorm3d(filter_nums[0], affine=affine),
                                         nn.Conv3d(filter_nums[0],num_classes, 1, stride=1, padding=0, bias=True)) 
        for res_idx in range(num_depths):
            if use_stem:
                self.stem_down[str(res_idx)] = nn.Sequential(
                    nn.Upsample(scale_factor=1/(2**res_idx), mode='trilinear', align_corners=True),
                    # Interpolate(scale_factor=1/(2**res_idx), mode='trilinear', align_corners=True),
                    nn.Conv3d(in_channels, filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx], affine=affine),
                    nn.ReLU(),
                    nn.Conv3d(filter_nums[res_idx],filter_nums[res_idx+1], 3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx+1], affine=affine)
                )       
                self.stem_up[str(res_idx)] = nn.Sequential(\
                                                    nn.ReLU(),
                                                    nn.Conv3d(filter_nums[res_idx+1],filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                                                    nn.InstanceNorm3d(filter_nums[res_idx], affine=affine),
                                                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))    
                
            else:
                self.stem_down[str(res_idx)] = nn.Sequential(
                    nn.Upsample(scale_factor=1/(2**res_idx), mode='trilinear', align_corners=True),
                    # Interpolate(scale_factor=1/(2**res_idx), mode='trilinear', align_corners=True),
                    nn.Conv3d(in_channels, filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx], affine=affine)
                )                
                self.stem_up[str(res_idx)] = nn.Sequential(\
                                                    nn.ReLU(),
                                                    nn.Conv3d(filter_nums[res_idx],filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                                                    nn.InstanceNorm3d(filter_nums[res_idx], affine=affine),
                                                    nn.Conv3d(filter_nums[res_idx],num_classes, 1),
                                                    nn.Upsample(scale_factor=2**res_idx, mode='trilinear', align_corners=True))  
                                       
        # define NAS search space
        if code is None:
            code_a = np.ones((num_blocks, len(code2out)))
            code_c = np.ones((num_blocks, len(code2out), cell_ops))
        else:
            code_a = code[1]
            code_c = F.one_hot(torch.from_numpy(code[2]), cell_ops).numpy()
        self.cell_tree = nn.ModuleDict()
        self.memory = np.zeros((num_blocks, len(code2out), cell_ops))
        for blk_idx in range(num_blocks):
            for res_idx in range(len(code2out)):
                if code_a[blk_idx, res_idx] == 1:
                    self.cell_tree[str((blk_idx,res_idx))] = \
                        cell(
                            filter_nums[code2in[res_idx] + int(use_stem)],
                            filter_nums[code2out[res_idx] + int(use_stem)], 
                            code2ops[res_idx],
                            # affine,
                            # k,
                            code_c[blk_idx, res_idx]
                        )
                    self.memory[blk_idx, res_idx] = np.array([_.memory + self.cell_tree[str((blk_idx,res_idx))].preprocess.memory
                                                             if _ is not None else 0 for _ in self.cell_tree[str((blk_idx,res_idx))].op._ops[:cell_ops]])
        # define cell and macro arhitecture probabilities    
        self.log_alpha_c = torch.nn.Parameter(torch.zeros(num_blocks, len(code2out), cell_ops)\
                                            .normal_(1, 0.01).cuda().requires_grad_())
        self.fix_c_grad_mask = torch.ones_like(self.log_alpha_c)
        self.log_alpha_a = torch.nn.Parameter(torch.zeros(num_blocks, len(code2out))\
                                            .normal_(0, 0.01).cuda().requires_grad_())
        self.fix_a_grad_mask = torch.ones_like(self.log_alpha_a)
        self._arch_param_names = ['log_alpha_a', 'log_alpha_c']

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]        
          
    def get_code_c(self, use_max=False):
        log_alpha = self.log_alpha_c
        if use_max:
            return torch.ones_like(log_alpha,requires_grad=False)
        code_c = torch.zeros_like(log_alpha,requires_grad=False)
        for blk_idx in range(self.num_blocks):
            for cell_idx in range(len(self.code2out)):
                code_c[blk_idx,cell_idx, np.random.randint(0, self.cell_ops)] = 1
        return code_c

    def get_code_a(self, k=2, use_max=False):
        """ Generate macro structure codes
            random sample k path per block
        """
        log_alpha = self.log_alpha_a
        device = log_alpha.device
        probs_a, code_prob_a = self.get_prob_a(child=False)
        if use_max:
            code_a = torch.ones((self.num_blocks, len(self.code2out)),requires_grad=False).to(device)    
            node_a = np.ones((self.num_blocks+1, self.num_depths)).astype(int)
            return node_a, code_a, code_prob_a      
        code_a = torch.zeros((self.num_blocks, len(self.code2out)),requires_grad=False).to(device)   
        node_a = np.zeros((self.num_blocks, self.num_depths)).astype(int)
        # init: random sample k path
        index = np.random.permutation(len(self.code2out))[:k]
        _path_activation = torch.zeros(len(self.code2out))
        _path_activation[index] = 1
        code_a[0] = _path_activation
        # convert init k path to node_activation
        _node_in = np.zeros(self.num_depths)
        _node_out = np.zeros(self.num_depths)
        for res_idx in range(len(self.code2out)):
            _node_out[self.code2out[res_idx]] += _path_activation[res_idx]
            _node_in[self.code2in[res_idx]] += _path_activation[res_idx]
        init = (_node_in >= 1).astype(int)
        node_activation = (_node_out >= 1).astype(int) # node_activation must be numpy array since it's used as dict key
        node_a[0] = node_activation
        for blk_idx in range(1, self.num_blocks):
            child_path = []
            child_node = []
            mtx = self.trans_mtx[str(node_activation)]
            node_idx = self.node_act_dict[str(node_activation)]
            for _ in range(len(mtx)):
                _node_activation = (np.matmul(mtx[_],node_activation)>=1).astype(int)
                _path_activation = torch.tensor(mtx[_].flatten()[self.tidx], requires_grad=False).to(device)
                child_node.append(_node_activation)
                child_path.append(_path_activation)
            # sample child model
            path_num = [sum(_).item() for _ in child_path]
            # in case k > max(path_num) e.g. k=3 while only a single node is activated
            index = np.where(np.array(path_num) == min(max(path_num), k))[0]              
            child_idx = np.random.permutation(index)[0]
            node_activation = child_node[child_idx]
            code_a[blk_idx,:] = child_path[child_idx]
            node_a[blk_idx,:] = node_activation
        return np.vstack([init,node_a]), code_a, code_prob_a

    def get_prob_a(self, child=False):
        """ Get final path probabilities and child model weights
        Args: return child probability as well (used in decode)
        """
        log_alpha = self.log_alpha_a
        _code_prob_a  = torch.sigmoid(log_alpha/self.ef) 
        norm = 1-(1-_code_prob_a).prod(-1) # normalizing factor
        code_prob_a = _code_prob_a/norm.unsqueeze(1)
        if child:
            probs_a = []
            path_activation = torch.from_numpy(self.child_list).cuda()
            for blk_idx in range(self.num_blocks):
                probs_a.append((path_activation * _code_prob_a[blk_idx] +\
                              (1 - path_activation) * (1 - _code_prob_a[blk_idx])).prod(-1) / norm[blk_idx] )
            probs_a = torch.stack(probs_a)
            return probs_a, code_prob_a
        else:
            return None, code_prob_a

    def mask_c(self, code_c, code_a):
        code_c_mask = copy.deepcopy(code_c)
        for blk_idx in range(code_a.shape[0]):
            for cell_idx in range(code_a.shape[1]):
                if code_a[blk_idx, cell_idx] == 0:
                    code_c_mask[blk_idx, cell_idx].fill_(0)
        return code_c_mask

    def update_ef(self, init_ef, iter_c, iter_t, scheme=0, end=1):
        """ Update annealing temperature (modified from early fix threshold in one-shot repo)
        init_ef: initial early fix threshold
        iter_c: current iteration
        iter_t: total iteration
        scheme:
        0: keep the same
        1: from init_ef to 1 at end/2*iter_t then decrease to zero at end*iter_t, only used for u-net initialization
        2: decrease from init_ef to 0 at end*iter_t
        3: cosine annealing with restarts
        4: cosine annealing
        end: decrease to zero at "end" percentage of total iteration
        """
        if scheme == 0:
            self.ef = init_ef
        elif scheme == 1: 
            if iter_c <= end/2*iter_t:
                self.ef = init_ef + (1-init_ef)*iter_c/(end/2*iter_t)
            else:
                self.ef = max(0, 2 - 2*iter_c/(end*iter_t))
        elif scheme == 2:
            self.ef = max(0.001, init_ef * max(0, 1 - iter_c/(end*iter_t)))            
        elif scheme == 3:
            self.ef = max(0.001, init_ef * ((1 + np.cos(np.pi * iter_c / (0.1*iter_t))) / 2) * ((1 + np.cos(np.pi * iter_c / iter_t)) / 2) )
        elif scheme == 4:
            self.ef = max(0.5 * init_ef * (1 + np.cos(np.pi * iter_c / iter_t)), 0.001)
    def get_memory_usage(self, in_size, full=False, cell_memory=False, code=None):
        """ Get estimated output tensor size
        in_size: input image shape at the highest resolutoin level
        full: full memory usage with all probability of 1
        """
        # convert input image size to feature map size at each level
        b, c, h, w, s = in_size
        sizes = []
        for res_idx in range(self.num_depths):
            sizes.append( b * self.filter_nums[res_idx] * h//(2**res_idx) * w//(2**res_idx) * s//(2**res_idx))
        sizes = torch.tensor(sizes).to(torch.float32).cuda()//(2**(int(self.use_stem)))
        probs_a, code_prob_a, = self.get_prob_a(child=False)
        cell_prob = F.softmax(self.log_alpha_c/self.ef, dim=-1)
        if full:
            code_prob_a = code_prob_a.detach()
            code_prob_a.fill_(1)
            if cell_memory:
                cell_prob = cell_prob.detach()
                cell_prob.fill_(1/self.cell_ops)
        memory = torch.from_numpy(self.memory).to(torch.float32).cuda()
        usage = 0
        for blk_idx in range(self.num_blocks):
            # node activation for input 
            # cell operation 
            for path_idx in range(len(self.code2out)):
                if code is not None:
                    usage += code[0][blk_idx, path_idx] * (1 + (memory[blk_idx, path_idx] * code[1][blk_idx, path_idx]).sum()) * sizes[self.code2out[path_idx]]
                else:
                    usage += code_prob_a[blk_idx, path_idx] * (1 + (memory[blk_idx, path_idx] * cell_prob[blk_idx, path_idx]).sum()) * sizes[self.code2out[path_idx]]
                

        return usage * 32 / 8 / 1024**2
    def get_topology_entropy(self, probs):
        """ Get topology entropy loss
        """
        if hasattr(self,'node2in'):
            node2in = self.node2in
            node2out = self.node2out
        else:
            # node activation index to feasible input child_idx
            node2in = [[] for i in range(len(self.node_act_list))] 
            # node activation index to feasible output child_idx
            node2out = [[] for i in range(len(self.node_act_list))]
            for child_idx in range(len(self.child_list)):
                _node_in, _node_out = np.zeros(self.num_depths), np.zeros(self.num_depths)
                for res_idx in range(len(self.code2out)):
                    _node_out[self.code2out[res_idx]] += self.child_list[child_idx][res_idx]
                    _node_in[self.code2in[res_idx]] += self.child_list[child_idx][res_idx]
                _node_in = (_node_in >= 1).astype(int)
                _node_out = (_node_out >= 1).astype(int)
                node2in[self.node_act_dict[str(_node_out)]].append(child_idx)
                node2out[self.node_act_dict[str(_node_in)]].append(child_idx)
            self.node2in = node2in
            self.node2out = node2out
        # calculate entropy
        ent = 0
        for blk_idx in range(self.num_blocks-1):
            blk_ent = 0
            # node activation probability
            for node_idx in range(len(self.node_act_list)):
                _node_p = probs[blk_idx, node2in[node_idx]].sum()        
                _out_probs = probs[blk_idx+1, node2out[node_idx]].sum()
                blk_ent += -(_node_p * torch.log(_out_probs + 1e-5)\
                            + (1-_node_p) * torch.log(1-_out_probs + 1e-5))            
            ent += blk_ent
        return ent

    def decode(self):
        """
        Decode network log_alpha_a/log_alpha_c using dijkstra shortpath algorithm
        Return:
            code with maximum probability
        """
        probs, code_prob_a = self.get_prob_a(child=True)
        code_a_max = self.child_list[torch.argmax(probs,-1).data.cpu().numpy()]
        code_c = torch.argmax(F.softmax(self.log_alpha_c/self.ef, -1),-1).data.cpu().numpy()
        probs = probs.data.cpu().numpy()
        # define adacency matrix
        amtx = np.zeros((1+len(self.child_list)*self.num_blocks+1, 1+len(self.child_list)*self.num_blocks+1))
        # build a path activation to child index searching dictionary
        path2child = {str(self.child_list[i]): i for i in range(len(self.child_list))}
        # build a submodel to submodel index
        sub_amtx = np.zeros((len(self.child_list),len(self.child_list)))
        for child_idx in range(len(self.child_list)):  
            _node_act = np.zeros(self.num_depths).astype(int)
            for path_idx in range(len(self.child_list[child_idx])):
                _node_act[self.code2out[path_idx]] += self.child_list[child_idx][path_idx]
            _node_act = (_node_act >= 1).astype(int)
            for mtx in self.trans_mtx[str(_node_act)]:
                connect_child_idx = path2child[str(mtx.flatten()[self.tidx].astype(int))]
                sub_amtx[child_idx, connect_child_idx] = 1
        # fill in source to first block, add 1e-5/1e-3 to avoid log0 and negative edge weights
        amtx[0, 1:1+len(self.child_list)] = -np.log(probs[0]+1e-5) + 0.001
        # fill in the rest blocks
        for blk_idx in range(1,self.num_blocks):
            amtx[1+(blk_idx-1)*len(self.child_list):1+blk_idx*len(self.child_list), \
                1+blk_idx*len(self.child_list):1+(blk_idx+1)*len(self.child_list)] \
                = sub_amtx * np.tile(-np.log(probs[blk_idx]+1e-5) + 0.001,(len(self.child_list),1))
        # fill in the last to the sink
        amtx[1+(self.num_blocks-1)*len(self.child_list):1+self.num_blocks*len(self.child_list),-1] = 0.001
        # solving shortest path problem
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        graph = csr_matrix(amtx)
        dist_matrix, predecessors, sources = dijkstra(csgraph=graph, directed=True, indices=0, min_only=True, return_predecessors=True)
        index, a_idx = -1, -1
        code_a = np.zeros((self.num_blocks, len(self.code2out)))
        node_a = np.zeros((self.num_blocks+1, self.num_depths))
        # decoding to paths
        while True:
            index = predecessors[index]
            if index == 0:
                break
            child_idx = (index - 1)%len(self.child_list)
            code_a[a_idx,:] = self.child_list[child_idx]
            for res_idx in range(len(self.code2out)):
                node_a[a_idx,self.code2out[res_idx]] += code_a[a_idx,res_idx]
            a_idx -= 1
        for res_idx in range(len(self.code2out)):
            node_a[a_idx,self.code2in[res_idx]] += code_a[0,res_idx]       
        node_a = (node_a >= 1).astype(int)
        return node_a, code_a, code_c, code_a_max

    def forward(self, x, code=None, ds=False):
        """ Prediction based on dynamic code
        Args:
            x: input tensor
            code: [node_a, code_a, code_c]
            ds: direct skip
        """
        # define output positions
        out_pos = [self.num_blocks-1]
        # sample path weights
        predict_all = []
        node_a, code_a, code_c = code
        probs_a, code_prob_a = self.get_prob_a(child=False)
        # stem inference
        inputs = []
        finefeat = 0
        for _ in range(self.num_depths):
            if node_a[0][_]:
                if _ == 0 and ds:
                    finefeat = self.stem_down[str(_)][:4](x)
                    inputs.append(self.stem_down[str(_)][4:](finefeat))
                else:
                    inputs.append(self.stem_down[str(_)](x))
            else:
                inputs.append(None)

        for blk_idx in range(self.num_blocks):
            outputs = [0]*self.num_depths
            for res_idx, activation in enumerate(code_a[blk_idx].data.cpu().numpy()):
                if activation:
                    if self.istrain:
                        outputs[self.code2out[res_idx]] += self.cell_tree[str((blk_idx,res_idx))]\
                                                                (inputs[self.code2in[res_idx]], ops=code_c[blk_idx,res_idx],
                                                                weight= F.softmax(self.log_alpha_c[blk_idx, res_idx]/self.ef, dim=-1))\
                                                                * code_prob_a[blk_idx, res_idx]
                    else:
                        outputs[self.code2out[res_idx]] += self.cell_tree[str((blk_idx,res_idx))]\
                                                                (inputs[self.code2in[res_idx]], ops=code_c[blk_idx,res_idx],
                                                                weight= torch.ones_like(code_c[blk_idx,res_idx],requires_grad=False))  
            inputs = outputs
            if blk_idx in out_pos:
                start = False
                for res_idx in range(self.num_depths-1,-1,-1):
                    if start:
                        _temp = self.stem_up[str(res_idx)](inputs[res_idx]+_temp)
                    elif node_a[blk_idx+1][res_idx]:
                        start = True
                        _temp = self.stem_up[str(res_idx)](inputs[res_idx])         
                prediction = self.stem_finals(finefeat+_temp)
                predict_all.append(prediction)
        return predict_all