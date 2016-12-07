import cPickle, gzip
import numpy as np
import time
import random
import theano
import theano.tensor as T
import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, MySeq2Seq
from keras.layers import Activation

import os
from os import listdir
from os.path import isfile, join

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import scipy.misc


import util

nb_epoch = 400
batch_size = 20
fix_len = 400
hid_dim = 200

def shuffle_mfcc_input(wavfiles,all_feats):
    ''' Shuffle both wav file list and input mfcc sequences

    # Arguments
        wavfiles: wav file list
        all_feats: input mfcc sequences of all the wav file

    # Returns
        Shuffled wavfiles & all_feats
    '''
    index_shuf = range(len(wavfiles))
    random.shuffle(index_shuf)
    wavfiles_shuf  = [ wavfiles[i] for i in index_shuf ]
    all_feats_shuf = [ all_feats[i] for i in index_shuf]
    wavfiles = wavfiles_shuf
    all_feats = all_feats_shuf

    return wavfiles, all_feats

def load_mfcc_train(wavfiles,cfgpath):
    ''' Load all the mfcc feats of the wav files fromm wavfiles(list)
    # Arguments
        wavfiles: wav file list
        cfgpath: cfg file path

    # Returns
        wavfiles: wav file list
        mfcc_feats: mfcc features of each wav file in wavfiles
    '''
    if isfile('mfcc_feats'):
        with open('mfcc_feats') as f:
            mfcc = cPickle.load(f)
            wavfiles = mfcc[0]
            mfcc_feats = mfcc[1]
    else:
        count = 0
        for f in wavfiles:
            feat = util.get_mfcc_feat(f,cfgpath)
            mfcc_feats.append(feat)
            count += 1

            print 'Process {} files...'.format(count)

        dump = [wavfiles,mfcc_feats]
        with open('mfcc_feats','w') as f:
            cPickle.dump(dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print 'Loaded mfcc_feats!'

    return wavfiles,mfcc_feats

def build_batch(i,nb_batch,wavfiles,mfcc_feats,utt2LabelSeq,label2id):
    batch_left = len(wavfiles) % batch_size
    if i == nb_batch-1:
       cur_batch_size = batch_size+batch_left
       X_train = np.zeros((batch_size + batch_left,fix_len,39))
       Y_train = np.zeros((batch_size + batch_left , fix_len, len(label2id)))
    else:
        cur_batch_size = batch_size
        X_train = np.zeros((batch_size,fix_len,39))
        Y_train = np.zeros((batch_size,fix_len,len(label2id)))

    ### paddling & truncate
    n = i*batch_size
    for b in range(cur_batch_size):
       f = os.path.basename(wavfiles[n])[:-4]
       if utt2LabelSeq[f].shape[0] > fix_len:
           Y_train[b] = utt2LabelSeq[f][:fix_len]
           X_train[b] = mfcc_feats[n][:fix_len]
       else:
           Y_train[b] = np.lib.pad(utt2LabelSeq[f],((0,fix_len-utt2LabelSeq[f].shape[0]),(0,0)),'constant',\
                   constant_values=0.)
           Y_train[b][:,label2id['sil']] = 1.
           X_train[b] = np.lib.pad(mfcc_feats[n],((0,fix_len-utt2LabelSeq[f].shape[0]),(0,0)),'constant',constant_values=0.)
       n += 1
    return X_train, Y_train

def build_sample_weight(y,label2id):
    ''' Return sample weight
    # Arguments:
        y: numpy array; shape: (None,len,nb_target)
        label2id: dict; map label(str) to id(int)
    # Return:
        sample_weight: numpy array; same shape as y
    '''
    sil = label2id['sil']
    sp = label2id['sp']

    y_id = y.argmax(axis=2)
    sample_weight = np.ones(y_id.shape)
    sample_weight[ y_id == sil ] = 0.
    sample_weight[ y_id == sp ] = 0.

    return sample_weight

def mnist_test():

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
    ### load mfcc input
    dicpath = '/home/troutman/lab/STD/Pattern/Result/timit_c50_s5_g1/library/dictionary.txt'
    mlfpath =  '/home/troutman/lab/STD/Pattern/Result/timit_c50_s5_g1/result/result.mlf'
    wavpath  = '/home/troutman/lab/timit/train/wav/'
    cfgpath  = '/home/troutman/lab/STD/Pattern/zrst/matlab/hcopy.cfg'

    mfcc_feats = [] # shape:(nb_wavfiles,nb_frame,39)
    wavfiles = [join(wavpath,f) for f in listdir(wavpath) if isfile(join(wavpath, f))]

    ### load mfcc feats
    wavfiles,mfcc_feats = load_mfcc_train(wavfiles,cfgpath)
   
    utt2LabelSeq,label2id,id2label = util.MLFReader(mlfpath,cfgpath,dicpath)

    ### build model
    model = MySeq2Seq(input_shape=(fix_len,39), output_dim=len(label2id),output_length=fix_len,hidden_dim=hid_dim)
    #model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = 'adadelta',sample_weight_mode="temporal")
    #model.compile(loss='mse', optimizer = 'rmsprop')
    print model.summary()

    nb_batch = int(len(wavfiles)/batch_size)
    for epoch in range(nb_epoch):
        #print 'Epoch: {:3d}'.format(epoch)
        start_time = time.time()

        ### for each target
        ### load y target
        utt2LabelSeq,label2id,id2label = util.MLFReader(mlfpath,cfgpath,dicpath)

        
        ### shuffle data
        wavfiles, mfcc_feats = shuffle_mfcc_input(wavfiles,mfcc_feats)

        ### train on batch
        batch_loss = 0.
        for i in range(nb_batch):
            #print '    batch: {:4d}'.format(i)
            X_train, Y_train = build_batch(i,nb_batch,wavfiles,mfcc_feats,utt2LabelSeq,label2id)

            ### build class weight(not train sil & sp)
            sample_weight = build_sample_weight(Y_train,label2id)
           
            ### for loop to train each decoder
            #print X_train.shape, Y_train.shape
            batch_loss += model.train_on_batch(X_train,Y_train,sample_weight=sample_weight)

        print 'Epoch {:3d} loss: {:.5f}  fininish in: {:.5f} sec'.format(epoch,batch_loss/nb_batch,time.time() - start_time)
        model.save_weights('ce_adadelta_classw_hid200.h')
        del utt2LabelSeq; del label2id;del id2label; del X_train; del Y_train;

    # show the result?


if __name__ == "__main__":
    ha()
