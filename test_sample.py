import numpy as np

def random_choice_noreplace(m,n, axis=-1):
    # m, n are the number of rows, cols of output
    return np.random.rand(m,n).argsort(axis=axis)


targets = np.random.choice(50000, 512*16).reshape(-1)
length = 50000

# base = np.tile(np.arange(length), (512, 1)) # 53ms

def random_sampling(length, items, num):
    token_num = targets.shape[-1]

    # items = np.copy(targets.view(-1))
    # import pdb
    # pdb.set_trace()
    # print(items.shape, np.unique(items).shape)
    vocab_num = length
    # out = np.array([np.concatenate([np.array([items[i]]), np.random.randint(0, items[i], int(items[i]/vocab_num*511)), np.random.randint(items[i]+1, vocab_num, 511-int(items[i]/vocab_num*511))]) for i in range(len(items))])
    strange = np.setdiff1d(np.arange(vocab_num), items)
    hav = np.unique(items)
    out = np.concatenate([hav, np.random.choice(strange, hav.shape[0], replace=False)])
    tab = np.ones(vocab_num,dtype=np.int32)*-1
    for idx,item in enumerate(out):
        tab[item] = idx
    for idx,item in enumerate(items):
        items[idx] = tab[item]

    # out = [np.concatenate([np.random.choice(targets[i], int(targets[i]/length*1024)), np.random.choice(length-targets[i], 1024-int(targets[i]/length*1024))+targets[i]+1]) for i in range(token_num)]
    # out = np.array([np.concatenate([np.array([targets[i]]), np.random.randint(0, targets[i], int(targets[i]/length*1023)), np.random.randint(targets[i]+1, length, 1023-int(targets[i]/length*1023))+targets[i]+1]) for i in range(token_num)])

    # base = base[:token_num,:]
    # # base = np.repeat(np.arange(length).reshape(1,-1), token_num, 0) #53.9ms
    # mask = np.ones((token_num, length), dtype=bool) # 1.2ms
    # mask[range(token_num), targets] = False #range(token_num)


    # out = base[mask].reshape(token_num, -1)    
    # out = out[np.arange(len(out))[:,None], np.random.randn(*out.shape).argsort(axis=1)][:, :num]
    
    return out

# random_sampling(length, targets, base, 1024)

# %timeit random_choice_noreplace(1000,100)

# item = dataset[0]
# vocab_num = len(self.source_dictionary)
# out = np.array([np.concatenate([np.array([item[i]]), np.random.randint(0, item[i], int(targets[i]/item*1023)), np.random.randint(targets[i]+1, vocab_num, 1023-int(targets[i]/50000*1023))+item[i]+1]) for i in range(len(item))])