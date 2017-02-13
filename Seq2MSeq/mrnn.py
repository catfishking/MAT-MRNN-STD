import cPickle, gzip
import numpy as np
import math
import time
import random
import sys
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

nb_epoch = 1000
batch_size = 40
fix_len = 50
bucket_size = 50
hid_dim = 100
phone_level = True
targets = ['timit_c50_s5_g1_phone','timit_c50_s3_g1_phone','timit_c50_s7_g1_phone',\
        'timit_c100_s5_g1_phone','timit_c100_s3_g1_phone','timit_c100_s7_g1_phone',\
        'timit_c300_s5_g1_phone','timit_c300_s3_g1_phone','timit_c300_s7_g1_phone']
buckets = [0,5,10,15,20,25,30,35,40,50,60,70,90,100,150]
weight_name = 'ce_adadelta_classw_hid100_phone_bucket50-test.h'

class WordAlign():
    def __init__(self,wavname,start,end,vol):
        self.wav = wavname
        self.start = start
        self.end = end
        self.vol = vol
        self.length = end - start

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

            print ('Process {} files...'.format(count))

        dump = [wavfiles,mfcc_feats]
        with open('mfcc_feats','w') as f:
            cPickle.dump(dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print ('Loaded mfcc_feats!')

    return wavfiles,mfcc_feats

def load_word_align(path, align_dict):
    ''' Load word_align info '''

    with open(path,'r') as f:
        for line in f:
            line = line.split() # line: dr1_asdf_sa1(wavfilename) 45(start) 65(end) hello(word)
            align = WordAlign(line[0],int(line[1]),int(line[2]),line[3])
            align_dict.setdefault(line[0],[]).append(align)
    print ('Load word align')

def build_batch(i,nb_batch,wavfiles,mfcc_feats,utt2LabelSeq,label2id,id2label,phone_level=False):
    ''' Build a batch
    # Arguments:
        i: i-th batch
        nb_batch: total number of batch
        wavfiles: wav file list
        mfcc_feats: mfcc features of each wav file in wavfiles
        utt2LabelSeq: dict; map wav filename to its pattern
        label2id: dict; map label(str) to id(int)
        id2label: dict; map id(int) to label(str)
        phone_level: bool; whether target on phone-level
    # Returns:
        X_train, Y_train
    '''
    batch_left = len(wavfiles) % batch_size
    if i == nb_batch-1 and batch_left != 0:
       cur_batch_size = batch_left
       X_train = np.zeros((batch_left,fix_len,39))
       Y_train = np.zeros((batch_left , fix_len, len(id2label)))
    else:
        cur_batch_size = batch_size
        X_train = np.zeros((batch_size,fix_len,39))
        Y_train = np.zeros((batch_size,fix_len,len(id2label)))

    ### paddling & truncate
    n = i*batch_size
    for b in range(cur_batch_size):
        f = os.path.basename(wavfiles[n])[:-4]
        if utt2LabelSeq[f].shape[0] > fix_len:
           Y_train[b] = utt2LabelSeq[f][:fix_len]
           X_train[b] = mfcc_feats[n][:fix_len]
        else:
            X_train[b] = np.lib.pad(mfcc_feats[n],((0,fix_len-utt2LabelSeq[f].shape[0]),(0,0)),'constant',constant_values=0.)
            Y_train[b] = np.lib.pad(utt2LabelSeq[f],((0,fix_len-utt2LabelSeq[f].shape[0]),(0,0)),'constant',constant_values=0.)
            if phone_level:
                Y_train[b][utt2LabelSeq[f].shape[0]:,label2id['<s>']] = 1.
            else:
                Y_train[b][utt2LabelSeq[f].shape[0]:,label2id['sil']] = 1.
        n += 1
    return X_train, Y_train

# TODO this is the vanilla version
def build_batch_bucket(i,nb_batch,wavfiles,mfcc_feats,utt2LabelSeq,label2id,id2label,bucket_size=50,phone_level=False):
    ''' Build a batch based on bucket size
    # Arguments:
        i: i-th batch
        nb_batch: total number of batch
        wavfiles: wav file list
        mfcc_feats: mfcc features of each wav file in wavfiles
        utt2LabelSeq: dict; map wav filename to its pattern
        label2id: dict; map label(str) to id(int)
        id2label: dict; map id(int) to label(str)
        bucket_size: size of the bucket
        phone_level: bool; whether target on phone-level
    # Returns:
        X_train, Y_train
    '''
    batch_left = len(wavfiles) % batch_size
    if i == nb_batch-1 and batch_left != 0:
        cur_batch_size = batch_left
    else:
        cur_batch_size = batch_size

    X_train = np.array([])
    Y_train = np.array([])

    ### paddling & truncate
    n = i*batch_size
    for b in range(cur_batch_size):
        f = os.path.basename(wavfiles[n])[:-4]
        utt_len = utt2LabelSeq[f].shape[0]
        utt_bucket_n = int(math.ceil(utt_len / float(bucket_size)))

        x = np.lib.pad(mfcc_feats[n],((0,utt_bucket_n*bucket_size-utt_len),\
                (0,0)),'constant',constant_values=0.)
        y = np.lib.pad(utt2LabelSeq[f],((0,utt_bucket_n*bucket_size-utt_len),\
                (0,0)),'constant',constant_values=0.)
        if phone_level:
            y[utt_len:,label2id['<s>']] = 1.
        else:
            y[utt_len:,label2id['sil']] = 1.
        ### convert x,y from 2d to 3d batch array
        x = x.reshape(utt_bucket_n,bucket_size,39)
        y = y.reshape(utt_bucket_n,bucket_size,len(id2label))
        if X_train.size == 0: ### check empty
            X_train = x
            Y_train = y
        else:
            X_train = np.append(X_train,x,axis=0)
            Y_train = np.append(Y_train,y,axis=0)
        n += 1
    return X_train, Y_train

def build_sample_weight(y,label2id,phone_level=True):
    ''' Return sample weight
    # Arguments:
        y: numpy array; shape: (None,len,nb_target)
        label2id: dict; map label(str) to id(int)
    # Return:
        sample_weight: numpy array; same shape as y
    '''

    y_id = y.argmax(axis=2)
    sample_weight = np.ones(y_id.shape)
    if phone_level:
        sample_weight[ y_id == label2id['<s>']] = 0.01
    else:
        sample_weight[ y_id == label2id['sil'] ] = 0.01
        sample_weight[ y_id == label2id['sp'] ] = 0.01

    return sample_weight

def build_sample_weight_axis1(y,label2id,phone_level=True):
    ''' Return sample weight
    # Arguments:
        y: numpy array; shape: (len,nb_target)
        label2id: dict; map label(str) to id(int)
    # Return:
        sample_weight: numpy array; same shape as y
    '''

    y_id = y.argmax(axis=1)
    sample_weight = np.ones(y_id.shape)
    if phone_level:
        sample_weight[ y_id == label2id['<s>']] = 0.01
    else:
        sample_weight[ y_id == label2id['sil'] ] = 0.01
        sample_weight[ y_id == label2id['sp'] ] = 0.01

    return sample_weight


def Mygenerator(wavfiles,mfcc_feats,word_aligns,utt2LabelSeq,label2id,id2label,bucket_size,phone_level=True):
    while 1:
        for n,wav in enumerate(wavfiles):
            wav = os.path.basename(wav)[:-4] # remove '.wav'
            for word in word_aligns[wav]:
                if word.length <= bucket_size[1] and word.length > bucket_size[0]:
                    x = mfcc_feats[n][word.start:word.end]
                    x = np.lib.pad(x,\
                            ((0,bucket_size[1] - x.shape[0]),(0,0)),\
                            'constant',constant_values=0.)
                    y = utt2LabelSeq[wav][word.start:word.end]
                    y = np.lib.pad(utt2LabelSeq[wav][word.start:word.end],\
                            ((0,bucket_size[1] - y.shape[0]),(0,0)),\
                            'constant',constant_values = 0.)
                    if phone_level:
                        y[word.length:,label2id['<s>']] = 1.
                    else:
                        y[word.length:,label2id['sil']] = 1.
                    x = x[np.newaxis,:]
                    y = y[np.newaxis,:]
                    sample_weight = build_sample_weight(y,label2id,phone_level=phone_level)
                    #print wav,word.vol,sample_weight
                    yield x, y, sample_weight

def seq2seq_load_weights(model, encoder_weights, decoder_weights):
    ''' Load weights of the seq2seq model
    # Arguments:
        model: seq2seq model
        encoder_weights: list; weights of model layer 0,1,2
        decoder_weights: list; weights of model layer -2,-1
    '''
    for i in range(3):
        model.layers[i].set_weights(encoder_weights[i])
    for i,j in zip(range(-2,0),range(2)):
        model.layers[i].set_weights(decoder_weights[j])

def seq2seq_save_weights(model, encoder_weights, decoder_weights):
    ''' Save weights of the seq2seq model
    # Arguments:
        model: seq2seq model
        encoder_weights: list; weights of model layer 0,1,2
        decoder_weights: list; weights of model layer -2,-1
    '''
    for i in range(3):
        encoder_weights[i] = model.layers[i].get_weights()
    for i,j in zip(range(-2,0),range(2)):
        decoder_weights[j] = model.layers[i].get_weights()


def mnist_test():

    # Load the dataset
    f = open('../../mnist.pkl', 'r')
    train_set, valid_set, test_set = cPickle.load(f)
    print (train_set[0].shape)

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
    target_prefix = '/data/home/troutman/lab/STD/Pattern/Result/'
    dic_suffix = 'library/dictionary.txt'
    mlf_suffix =  'result/result.mlf'
    wavpath  = '/data/home/troutman/lab/timit/train/wav/'
    cfgpath  = '/data/home/troutman/lab/STD/Pattern/zrst/matlab/hcopy.cfg'
    word_align_path = '/data/home/troutman/lab/timit/train/word_all.txt'
    weight_dir = 'model_weights'

    ### check save weight file exist or not
    if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
    if os.path.isfile(join(weight_dir,weight_name)):
        stdin = raw_input('overwrite existed file {} ?(y,n)'.format(weight_name))
        if 'n' in stdin:
            sys.exit(1)

    mfcc_feats = [] # shape:(nb_wavfiles,nb_frame,39)
    wavfiles = [join(wavpath,f) for f in listdir(wavpath) if isfile(join(wavpath, f))]

    ### load mfcc feats
    wavfiles,mfcc_feats = load_mfcc_train(wavfiles,cfgpath)

    ### load word_align
    word_aligns = {}
    load_word_align(word_align_path,word_aligns)

    encoder_weights = []
    decoder_weights = [] # store layer[-2], layer[-1] weights for each target decoder
    ### get init weights
    for t in targets:
        print (t)
        mlfpath = join(join(target_prefix,t), mlf_suffix)
        dicpath = join(join(target_prefix,t), dic_suffix)
        utt2LabelSeq,label2id,id2label = util.MLFReader(mlfpath,cfgpath,dicpath,phone_level = phone_level)
        ### build model
        model = MySeq2Seq(input_shape=(fix_len,39), output_dim=len(id2label),output_length=fix_len,hidden_dim=hid_dim)
        #model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer = 'adadelta',sample_weight_mode="temporal")
        #model.compile(loss='mse', optimizer = 'rmsprop')
        model.summary()
        decoder_weights.append( [model.layers[-2].get_weights(), model.layers[-1].get_weights()] )
        encoder_weights = [model.layers[0].get_weights(), model.layers[1].get_weights(), model.layers[2].get_weights()]

    #nb_batch = int(math.ceil(len(wavfiles)/float(batch_size)))
    for epoch in range(nb_epoch):
        print 'Epoch: {:3d}'.format(epoch)
        start_time = time.time()
        epoch_loss = 0.
        ### for each target
        for t,n_t in zip(targets,range(len(targets))):
            print 'Target: {}'.format(t)
            t_start_time = time.time()

            mlfpath = join(join(target_prefix,t), mlf_suffix)
            dicpath = join(join(target_prefix,t), dic_suffix)

            ### load y target
            utt2LabelSeq,label2id,id2label = util.MLFReader(mlfpath,cfgpath,dicpath, phone_level = phone_level)

            for bs in range(1,len(buckets)):
                print 'bucket: {}'.format(buckets[bs])
                ### build model
                model = MySeq2Seq(input_shape=(buckets[bs],39), output_dim=len(id2label),output_length=buckets[bs],hidden_dim=hid_dim)
                model.compile(loss='categorical_crossentropy', optimizer = 'adadelta',sample_weight_mode="temporal")
                ### load weights
                seq2seq_load_weights(model, encoder_weights, decoder_weights[n_t])
            
                ### shuffle data
                #wavfiles, mfcc_feats = shuffle_mfcc_input(wavfiles,mfcc_feats)
                ### train on batch
                batch_loss = 0.
                #for i in range(nb_batch):
                    #print '    batch: {:4d}'.format(i)
                    #X_train, Y_train = build_batch_bucket(i,nb_batch,wavfiles,mfcc_feats,\
                    #        utt2LabelSeq,label2id,id2label,phone_level=phone_level,bucket_size=bucket_size)
                    ### build class weight(not train sil & sp)
                    #sample_weight = build_sample_weight(Y_train,label2id,phone_level=phone_level)
                    ### for loop to train each decoder
                    #batch_loss += model.train_on_batch(X_train,Y_train,sample_weight=sample_weight)
                history = model.fit_generator(Mygenerator(wavfiles,mfcc_feats,word_aligns,utt2LabelSeq,\
                                            label2id,id2label,(buckets[bs-1],buckets[bs])),\
                                            samples_per_epoch=4000, nb_epoch=1)
            #epoch_loss += history.history['loss']
            seq2seq_save_weights(model, encoder_weights, decoder_weights[n_t])
            #print ('  Epoch {:3d} target:[{}] loss: {:.5f}  fininish in: {:.5f} sec'.\
            #        format(epoch,t,batch_loss/nb_batch,time.time() - t_start_time))

        print ('Epoch {:3d}  fininish in: {:.5f} sec'.format(epoch,time.time() - start_time))
        model.save_weights(join(weight_dir,weight_name))
        #del utt2LabelSeq; del label2id;del id2label; del X_train; del Y_train;

    # show the result?


if __name__ == "__main__":
    ha()
