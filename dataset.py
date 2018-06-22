import tensorflow as tf
import numpy as np
import random
import math
    
class Dataset():
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
    
    def gen_next_batch(self, batch_size, is_train_set, epoch=None, iteration=None):
        if is_train_set==True:
            x = self.train_x
            y = self.train_y
        else:
            x = self.test_x
            y = self.test_y
            
        assert len(x)>=batch_size, "batch size must be smaller than data size {}.".format(len(x))
        
        if epoch != None:
            until = math.ceil(float(epoch*len(x))/float(batch_size))
        elif iteration != None:
            until = iteration
        else:
            assert False, "epoch or iteration must be set."
        
        iter_ = 0
        index_list = [i for i in range(len(x))]
        while iter_ <= until:
            idxs = random.sample(index_list, batch_size)
            iter_ += 1
            yield (x[idxs], y[idxs], idxs)
        
    
class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        (self.train_x, self.train_y), (self.test_x, self.test_y) = tf.keras.datasets.mnist.load_data()
        self.train_x = self.train_x.reshape(-1, self.train_x.shape[1]*self.train_x.shape[2])
        self.train_x = self.train_x*0.02
        self.test_x = self.test_x.reshape(-1, self.test_x.shape[1]*self.test_x.shape[2])
        self.test_x = self.test_x*0.02
        self.num_classes = 10
        self.feature_dim = 784