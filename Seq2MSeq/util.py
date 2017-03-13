import numpy as np
import subprocess
from subprocess import PIPE, Popen
import re

def label_id_builder(dic_path,label2id, id2label,phone_level):
    count = 0
    with open(dic_path,'r') as f:
        for line in f:
            line = line.split()
            if phone_level:
                if line[0] == 'sp' or line[0] == 'sil':
                    continue
            label2id[ line[0] ] = count
            id2label[count] = line[0]
            count += 1
    if phone_level:
        label2id['<s>'] = count
        label2id['</s>'] = count
        id2label[count] = '<s>'

        label2id['pad'] = count+1
        id2label[count+1] = 'pad'

def to_onehot(y,nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y:  class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
	A binary matrix representation of the input.
    '''

    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def MLFReader(mlf_path,cfg_path,dic_path,frame_level=True,phone_level=False):
    '''Convert MLF file into label sequence

     # Arguments
         mlf_path: path to .mlf file
         cfg_path: path to .cfg file
         dic_path: path to dictionary
         frame_level: return label sequences of each frame or not
         phone_level: target is phone-level or not

    # Returns
        utt to label_sequence(onehot encoding) dictionary,
        label2id,
        id2label    
    '''         

    target_rate = 0.
    # read cfg file to check the target_rate
    with open(cfg_path,'r') as f:
        for line in f:
            if 'TARGETRATE' in line:
                line = line.split('=')
                target_rate = float(line[1])

    label_seq = {} # map utt to label sequence
    label2id = {} # map label to label id
    id2label = {}

    # build label2id
    label_id_builder(dic_path,label2id,id2label,phone_level)

    utt = ''
    utt_label = []

    # read mlf file to generate label sequence
    with open(mlf_path) as f:
        for line in f:
            if line[0] == '.': # if is end of an utterance
                label_seq[utt] = to_onehot(utt_label,nb_classes=len(id2label))
                utt = ''
                utt_label = []

            elif line[0] == '\"' and line[-2] == '\"': # if is a new utterance
                utt = line[3:-6] # remove file extension

            elif len(utt) > 0:
                line = line.split()
                start = float(line[0])
                end = float(line[1])
                num_frame = int((end - start)/target_rate)
                label = label2id[ line[2] ]

                if frame_level:
                    for i in range(num_frame):
                        utt_label.append(label)
    return label_seq, label2id, id2label
                
def get_mfcc_feat(wav_path,config_path):
    '''Return mfcc feature based on config
    # Arguments
        wav_path: wav file path
        config_path: htk config file path
    # Returns
        A matrix of MFCC features
    '''

    p = Popen(['HList','-C',config_path,wav_path],stdout=PIPE)

    # read the MFCC features
    p.stdout.readline() # skip first line
    mfcc = np.array([])
    f_mfcc = np.zeros(39)
    count = 0
    for line in p.stdout:
        if ':' in line: # a new frame start
            f_mfcc = np.zeros(39)
            count = 0
            line = line.split()[1:]
            for feat in range(len(line)):
                f_mfcc[count] = line[feat]
                count += 1
        elif 'END' in line: # end of MFCC feats
            break
        else:
            line = line.split()
            for feat in range(len(line)):
                f_mfcc[count] = line[feat]
                count += 1

        if count == 39:
            mfcc = np.append(mfcc,f_mfcc,axis=0)

    p.wait() 

    mfcc = mfcc.reshape(mfcc.size/39,39)
    return mfcc

def get_word_mfcc(word_trans_path, mfcc, wav_fs=16000, mfcc_inv=10, use_pickle=False):
    """
    Arguments:
        word_trans_path : path of word-level transcript
        mfcc : mfcc vector
        wac_fs : sample rate of wav file , default = 16000
        mfcc_inv : mfcc interval , default = 10ms
    
    Return:
        word_mfcc: list of word's mfcc in the utterance
        word_list: list of words in the utterance
    """

    if use_pickle == True:
        paths = re.split("/|_", word_trans_path)
        print paths
        word_path = "/data/home/troutman/lab/timit/train/%s/%s/%s.wrd" \
                %(paths[-3], paths[-2], paths[-1][:-4])
        #print "word_path = ", word_path
        with open(word_path, "r") as f:
             trans = f.readlines()
    else: 
        with open(word_trans_path, "r") as f:
            trans = f.readlines()
    word_mfcc = []
    word_list = []
    for word in trans:
        word = word.split()
        start_point = int((float(word[0])/wav_fs) / (mfcc_inv*0.001))
        end_point = int((float(word[1])/wav_fs) / (mfcc_inv*0.001))
        #print "SE=", start_point, end_point
        tmp_word_mfcc = mfcc[start_point:end_point]

        word_mfcc.append(tmp_word_mfcc)
        word_list.append(word[2])
     
    return word_mfcc, word_list

def load_wrd_pos(wrd_path):
    class Utt_pos():
        def __init__(self,utt,wrd,start,end):
            self.utt = utt
            self.wrd = wrd
            self.start = int(float(start)/16000. / 0.01)
            self.end = int(float(end)/16000. /0.01)

    wrd2utt_pos = {}
    utt2wrd = {}

    with open(wrd_path,'r') as f:
        for line in f:
            if '.wrd' in line:
                utt = line[:-5]
            elif line.strip(): # not empty string
                line = line.split()
                
                start = line[0]
                end = line[1]
                wrd = line[2]
                utt_pos = Utt_pos(utt,wrd,start,end)
                wrd2utt_pos.setdefault(wrd,[]).append(utt_pos)
                utt2wrd.setdefault(utt,[]).append(wrd)
    return wrd2utt_pos, utt2wrd


def debug3():
    wrd_path = '../../timit/train/word_all.txt'
    wrd2utt_pos = load_wrd_pos(wrd_path)
    print len(wrd2utt_pos)
    #print [(key,len(wrd2utt_pos[key])) for key in wrd2utt_pos]

    wrd_path2 = '../../timit/test/word_all.txt'
    wrd2utt_pos2 = load_wrd_pos(wrd_path2)
    print len(wrd2utt_pos2)
    #print [(key,len(wrd2utt_pos2[key])) for key in wrd2utt_pos2]
    count = 0
    for key in wrd2utt_pos2:
        if key in wrd2utt_pos:
            count += 1
    print count


def debug():
    target = '../Pattern/Result/timit_c50_g1_s5/'
    mlf = target + 'result/result.mlf'
    cfg = target + 'library/config.cfg'
    dic = target + 'library/dictionary.txt'
    MLFReader(mlf,cfg,dic)

def debug2():
    from dtw import dtw
    from numpy.linalg import norm
    wav = '/tmp3/troutman/lab/timit/train/wav/dr1_fcjf0_sa1.wav'
    config = '/tmp3/troutman/lab/STD/Pattern/zrst/matlab/hcopy.cfg'
    mfcc1 = get_mfcc_feat(wav,config)
    w_mfcc1, w_list1 =get_word_mfcc('../../timit/train/dr1/fcjf0/sa1.wrd',mfcc1)

    wav = '/tmp3/troutman/lab/timit/train/wav/dr1_fcjf0_sa1.wav'
    config = '/tmp3/troutman/lab/STD/Pattern/zrst/matlab/hcopy.cfg'
    mfcc2 = get_mfcc_feat(wav,config)
    w_mfcc2, w_list2 =get_word_mfcc('../../timit/train/dr1/fcjf0/sa1.wrd',mfcc2)
    dist = dtw(w_mfcc1[0],w_mfcc2[1],dist=lambda x, y: norm(x - y, ord=1) )
    print w_mfcc1[0].shape, w_mfcc2[2].shape
    print dist


if __name__ == "__main__" :
    debug2()
