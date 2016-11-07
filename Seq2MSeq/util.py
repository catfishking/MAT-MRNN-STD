import numpy as np
import subprocess
from subprocess import PIPE, Popen

def label_id_builder(dic_path,label2id, id2label):
    count = 0
    with open(dic_path,'r') as f:
        for line in f:
            line = line.split()
            label2id[ line[0] ] = count
            id2label[count] = line[0]
            count += 1

def MLFReader(mlf_path,cfg_path,dic_path,frame_level=True):
    '''Convert MLF file into label sequence

     # Arguments
         mlf_path: path to .mlf file
         cfg_path: path to .cfg file
         dic_path: path to dictionary
         frame_level: return label sequences of each frame or not

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
    label_id_builder(dic_path,label2id,id2label)

    utt = ''
    utt_label = []

    # read mlf file to generate label sequence
    with open(mlf_path) as f:
        for line in f:
            if line[0] == '.': # if is end of an utterance
                label_seq[utt] = to_onehot(utt_label,nb_classes=len(label2id))
                utt = ''
                utt_label = []

            elif line[0] == '\"' and line[-2] == '\"': # if is a new utterance
                utt = line[3:-4] # remove file extension

            elif len(utt) > 0:
                line = line.split()
                start = float(line[0])
                end = float(line[1])
                num_frame = int((end - start)/target_rate)

                label = label2id[ line[2] ]

                if frame_level:
                    for i in range(num_frame):
                        utt_label.append(label)
                

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


def debug():
    target = '../Pattern/Result/timit_c50_g1_s5/'
    mlf = target + 'result/result.mlf'
    cfg = target + 'library/config.cfg'
    dic = target + 'library/dictionary.txt'
    MLFReader(mlf,cfg,dic)

def debug2():
    wav = '/tmp2/b02902077/timit/train/wav/dr8_mtcs0_sx82.wav'
    config = '/tmp2/b02902077/STD/Pattern/zrst/matlab/hcopy.cfg'
    mfcc = get_mfcc_feat(wav,config)
    print mfcc


if __name__ == "__main__" :
    debug2()
