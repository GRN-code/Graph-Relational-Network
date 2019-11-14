import numpy as np
import random

def get_indice_graph(adj, mask, size, keep_r=1.0):
    indices = mask.nonzero()[0]
    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    pre_indices = set()
    indices = set(indices)
    other = []
    for i in range(2708):
        other.append(i)
    other = set(other)
    
    indices = set(indices)  #一般不会改变  train还是140
    can_add = other - indices
    while len(indices) < size:
        new_add = indices - pre_indices
        if not new_add:
            break
        pre_indices = indices
        candidates = get_candidates(adj, new_add) - indices
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    if len(indices) < size:
        can_add = can_add - indices
        candidates = set(np.random.choice(list(can_add), size-len(indices), False))
        indices.update(candidates)
    print('indices size:-------------->', len(indices))
    return sorted(indices)

def get_indice_graph_val(adj, mask, size, keep_r=1.0):
    indices = mask.nonzero()[0]  # train 140 val 500 test 1000
   
    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    
    '''
    last_indices = set()
    if len(indices) > size:
        slice1 = random.sample(indices, size)
        # print(len(slice1))
        last_indices = slice1
        indices = last_indices
        # print(len(last_indices))
        # exit()
    # if len(indices) > size:
    '''
    batch = 175
    other = []
    for i in range(2708):
        other.append(i)
    other = set(other)
    
    sorte_indices = []
    indices = set(indices)  #一般不会改变  train还是140
    can_add = other - indices
    for batches in range(len(indices)//batch+1):
        if (batches+1)*batch > len(indices):
            indices_ = set(list(indices)[batches*batch:])
        else:
            indices_ = set(list(indices)[batches*batch:(batches+1)*batch])
        # print(indices_)
        pre_indices = set()
        print('***************************')

        while len(indices_) < size:
            new_add = indices_ - pre_indices
            if not new_add:
                break
            pre_indices = indices_
            # print(len(get_candidates(adj, new_add))) #数据集中有连接到的节点的index  如X=[0,2,3,0]  那么 1 2 节点是有被连接到的  get_...  [1,2]
            candidates = get_candidates(adj, new_add) - indices_
            if len(candidates) > size - len(indices_):
                candidates = set(np.random.choice(list(candidates), size-len(indices_), False))
            indices_.update(candidates)
        # print(indices)
        if len(indices_) < size:
            can_add = can_add - indices_
            candidates = set(np.random.choice(list(can_add), size-len(indices_), False))
            indices_.update(candidates)
        print('indices size:-------------->', len(indices_))
        sorte_indices.append(sorted(indices_))

    return sorte_indices


def get_indice_graph_test(adj, mask, size, keep_r=1.0):
    indices = mask.nonzero()[0]  # train 140 val 500 test 1000

    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    batch = 110
    other = []
    for i in range(19717):
        other.append(i)
    other = set(other)
    
    sorte_indices = []
    indices = set(indices)  #一般不会改变  train还是140
    can_add = other - indices
    for batches in range(len(indices)//batch+1):
        if (batches+1)*batch > len(indices):
            indices_ = set(list(indices)[batches*batch:])
        else:
            indices_ = set(list(indices)[batches*batch:(batches+1)*batch])
        # print(indices_)
        pre_indices = set()
        print('***************************')

        while len(indices_) < size:
            new_add = indices_ - pre_indices
            if not new_add:
                break
            pre_indices = indices_
            # print(len(get_candidates(adj, new_add))) #数据集中有连接到的节点的index  如X=[0,2,3,0]  那么 1 2 节点是有被连接到的  get_...  [1,2]
            candidates = get_candidates(adj, new_add) - indices_
            if len(candidates) > size - len(indices_):
                candidates = set(np.random.choice(list(candidates), size-len(indices_), False))
            indices_.update(candidates)
        # print(indices)
        if len(indices_) < size:
            can_add = can_add - indices_
            candidates = set(np.random.choice(list(can_add), size-len(indices_), False))
            indices_.update(candidates)
        print('indices size:-------------->', len(indices_))
        sorte_indices.append(sorted(indices_))

    return sorte_indices


def get_sampled_index(adj, size, center_num=1):
    n = adj.shape[0]
    pre_indices = set()
    indices = set(np.random.choice(n, center_num, False))
    while len(indices) < size:
        if len(pre_indices) != len(indices):
            new_add = indices - pre_indices
            pre_indices = indices
            candidates = get_candidates(adj, new_add) - indices
        else:
            candidates = random_num(n, center_num, indices)
        sample_size = min(len(candidates), size-len(indices))
        if not sample_size:
            break
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    return sorted(indices)


def get_candidates(adj, new_add):
    return set(adj[sorted(new_add)].sum(axis=0).nonzero()[1])


def random_num(n, num, indices):
    cans = set(np.arange(n)) - indices
    num = min(num, len(cans))
    if len(cans) == 0:
        return set()
    new_add = set(np.random.choice(list(cans), num, replace=False))
    return new_add
