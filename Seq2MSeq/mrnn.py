import cPickle, gzip
import numpy as np
import theano
import theano.tensor as T
import seq2seq
from seq2seq.models import SimpleSeq2Seq

import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc


import util


def main():

    # Load the dataset
    f = open('../../mnist.pkl', 'r')
    train_set, valid_set, test_set = cPickle.load(f)
    print train_set[0].shape

    model = SimpleSeq2Seq(input_dim=28, output_dim=28,output_length=28)
    model.compile(loss='mse', optimizer = 'rmsprop')
    X = train_set[0].reshape(50000,28,28)
#    a =  model.layers[0].get_weights()
#   print model.layers[0].set_weights(a)
    plt.imshow(X[0])
    plt.savefig('infig')
    model.fit(X,X)
    aa = model.predict(X)
    plt.imshow(aa[0])
    plt.savefig('outfig')

    # model.layers
    
def ha():
    fix_len = 550
    ### load mfcc input
    dicpath = '/tmp2/b02902077/STD/Pattern/Result/timit_c50_s5_g1/library/dictionary.txt'
    mlfpath =  '/tmp2/b02902077/STD/Pattern/Result/timit_c50_s5_g1/result/result.mlf'
    wavpath  = '/tmp2/b02902077/timit/train/wav/'
    cfgpath  = '/tmp2/b02902077/STD/Pattern/zrst/matlab/hcopy.cfg'

    all_feats = []
    wavfiles = [join(wavpath,f) for f in listdir(wavpath) if isfile(join(wavpath, f))]
    input_mfcc = np.zeros((len(wavfiles),fix_len,39))

    ### load mfcc feats
    if isfile('mfcc_feats'):
        with open('mfcc_feats') as f:
            mfcc = cPickle.load(f)
            wavfiles = mfcc[0]
            all_feats = mfcc[1]
    else:
        count = 0
        for f in wavfiles:
            feat = util.get_mfcc_feat(f,cfgpath)
            all_feats.append(feat)

            '''
            if feat.shape[0] > fix_len:
                feat = feat[:300]
            else:
                feat = np.lib.pad(feat,((0,fix_len-feat.shape[0]),(0,0)),'constant',constant_values=0)
            
            input_mfcc[count] = feat
            '''
            count += 1
            

            print 'Process {} files...'.format(count)

        dump = [wavfiles,all_feats]
        with open('mfcc_feats','w') as f:
            cPickle.dump(dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print 'Loaded mfcc_feats!'
        
    # load y target
    utt2LabelSeq,label2id,id2label = util.MLFReader(mlfpath,cfgpath,dicpath)
    
    # sort mfcc & y & paddling & truncate
    X_train = np.zeros((len(utt2LabelSeq),fix_len,39))
    Y_train = np.zeros((len(utt2LabelSeq),fix_len,len(label2id)))

    n = 0
    for f in wavfiles:
        f = os.path.basename(f)[:-4]
        if utt2LabelSeq[f].shape[0] > fix_len:
            Y_train[n] = utt2LabelSeq[f][:fix_len]
            X_train[n] = all_feats[n][:fix_len]
        else:
            Y_train[n] = np.lib.pad(utt2LabelSeq[f],((0,fix_len-utt2LabelSeq[f].shape[0]),(0,0)),'constant',constant_values=0.)
            X_train[n] = np.lib.pad(all_feats[n],((0,fix_len-utt2LabelSeq[f].shape[0]),(0,0)),'constant',constant_values=0.)
        n += 1
    
    # for loop to train each decoder
    X_batch = X_train[:1]
    Y_batch = Y_train[:1]
    model = SimpleSeq2Seq(input_shape=(550,39), output_dim=len(label2id),output_length=550)
    model.compile(loss='mse', optimizer = 'rmsprop')
    model.fit(X_batch,Y_batch,nb_epoch=100)
    model.save_weights('my_weights.h')

    # show the result?


if __name__ == "__main__":
    ha()
