import numpy as np
#import tensorflow as tf

# Read reverse syntax tree (string) and encode it
# Characteres: +,-,/,*,0,1,2,3,4,5,6,7,8,9
# Charactere embeddings: 14-dimensional one-hot vectors
# Ex.: + 5 * 2 3 --> 3 2 * 5 +

cons = ['0','1']
funcs = ['+','-','*','/']
symb = cons+funcs

char_id = {'0':0,'1':1,'+':2,'-':3,'*':4,'/':5,'#':6,'.':7}
all_symbs = symb + ['#','.']

num_chars = 8
max_level = 5
max_seq = 2**(max_level+1)

def gen_tree(d,h):
    if d == h:
        return '#'
    else:
        op = np.random.choice(symb,p=[0.05,0.05,0.225,0.225,0.225,0.225])
        if op in funcs:
            return op + gen_tree(d+1,h) + gen_tree(d+1,h)
        else:
            return op

def string2onehot(s):
    s += '.'
    onehot = np.zeros((max_seq,num_chars)) # len(s)
    for i in range(len(s)):
        c_id = char_id[s[i]]
        onehot[i,c_id] = 1.0
    return onehot

def onehot2string(oh):
    s = ''
    for c in oh:
        if np.argmax(c) == char_id['.']:
            break
        s += all_symbs[np.argmax(c)]
    return s

def get_batch(batch_size):
    return np.array([string2onehot(gen_tree(0,max_level)) for i in range(batch_size)])

