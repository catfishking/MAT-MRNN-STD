import os
import struct
import numpy as np
# import cPickle as pickle

def make_dumpfile(wav_dir, cluster_number, dump_file):
    from zrst.m_path import matlab_path
    import sys
     
    os.system('bash {}/run_clusterDetection.sh {} {} {}'.format(matlab_path, wav_dir, cluster_number, dump_file))

def rand_dumpfile(dump_file,cluster_number,rand_file):
    import random
    text = open(dump_file,'r').readlines()
    A = open(rand_file,'w')
    for i in range(len(text)):
        if '.wav' in text[i]:
            A.write(text[i])
        else:
            A.write(str(random.randint(1,cluster_number))+'\n')

def flat_dumpfile(dump_file,flat_file):
    I = open(dump_file,'r')
    O = open(flat_file,'w')
    for line in I:
        if 'wav' in line:
            O.write(line)
        else:
            for p in line.strip('\n').split():
                if p: O.write(p+'\n')



class DTW(object):
    def __init__(self, seq1, seq2, distance_func=None):
        '''
        seq1, seq2 are two lists,
        distance_func is a function for calculating
        the local distance between two elements.
        '''
        self._seq1 = seq1
        self._seq2 = seq2
        self._distance_func = distance_func if distance_func else lambda: 0
        self._map = {(-1, -1): 0.0}
        self._distance_matrix = {}
        self._path = []

    def get_distance(self, i1, i2):
        ret = self._distance_matrix.get((i1, i2))
        if not ret:
            ret = self._distance_func(self._seq1[i1], self._seq2[i2])
            self._distance_matrix[(i1, i2)] = ret
        return ret

    def calculate_backward(self, i1, i2):
        '''
        Calculate the dtw distance between
        seq1[:i1 + 1] and seq2[:i2 + 1]
        '''
        if self._map.get((i1, i2)) is not None:
            return self._map[(i1, i2)]

        if i1 == -1 or i2 == -1:
            self._map[(i1, i2)] = float('inf')
            return float('inf')

        min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                             key=lambda x: self.calculate_backward(*x))

        self._map[(i1, i2)] = self.get_distance(i1, i2) + \
                              self.calculate_backward(min_i1, min_i2)

        return self._map[(i1, i2)]

    def get_path(self):
        '''
        Calculate the path mapping.
        Must be called after calculate()
        '''
        i1, i2 = (len(self._seq1) - 1, len(self._seq2) - 1)
        while (i1, i2) != (-1, -1):
            self._path.append((i1, i2))
            min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                                 key=lambda x: self._map[x[0], x[1]])
            i1, i2 = min_i1, min_i2
        return self._path

    def calculate(self):
        return self.calculate_backward(len(self._seq1) - 1,
                                       len(self._seq2) - 1)


class SubDTW(DTW):
    def __init__(self, seq1, seq2, distance_func=None):
        '''
        seq1, seq2 are two lists,
        distance_func is a function for calculating
        the local distance between two elements.
        '''
        DTW.__init__(self, seq1, seq2, distance_func)

        self._map = {}
        for ___ in range(-1, len(seq1)):
            self._map[(___, -1)] = 0.0
        self.end_pos = -1
        self.beg_pos = -1

    def calculate_backward(self, i1, i2):
        '''
        Calculate the dtw distance between
        seq1[:i1 + 1] and seq2[:i2 + 1]
        '''
        if self._map.get((i1, i2)) is not None:
            return self._map[(i1, i2)]

        if i1 == -1:
            self._map[(i1, i2)] = float('inf')
            return float('inf')

        min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                             key=lambda x: self.calculate_backward(*x))

        self._map[(i1, i2)] = self.get_distance(i1, i2) + \
                              self.calculate_backward(min_i1, min_i2)

        return self._map[(i1, i2)]

    def get_path(self):
        '''
        Calculate the path mapping.
        Must be called after calculate()
        '''
        i1, i2 = (self.end_pos, len(self._seq2) - 1)
        while i2 != -1:
            self._path.append((i1, i2))
            min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                                 key=lambda x: self._map[x[0], x[1]])
            i1, i2 = min_i1, min_i2
        self.beg_pos = self._path[-1][0]
        return self._path

    def calculate(self):
        self.calculate_backward(len(self._seq1) - 1,
                                len(self._seq2) - 1)
        self.end_pos = min(range(-1, len(self._seq1)), key=lambda x: self._map[(x, len(self._seq2) - 1)])
        return self._map[self.end_pos, len(self._seq2) - 1]


'''
#debug DTW
a = [0,1,2,3,4,5,6,7]
b = [2,3,4,5]
A = DTW(a,b,lambda x,y:abs(x-y))
print A.calculate()
print A.get_path()

B = SubDTW(a,b,lambda x,y:abs(x-y))
print B.calculate()
print B.get_path()
'''


def read_feature(file):
    fin = open(file, 'rb')
    nN = struct.unpack('<i', fin.read(4))[0]
    period = struct.unpack('<i', fin.read(4))[0]
    nF = struct.unpack('<h', fin.read(2))[0] / 4
    ftype = struct.unpack('<h', fin.read(2))[0]
    fmat = np.zeros([nN, nF])
    for i in range(nN):
        for j in range(nF):
            fmat[i, j] = struct.unpack('<f', fin.read(4))[0]
    print nN, nF
    return fmat
'''
from pylab import *
file = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_MFCC/N200108011200-01-01.mfc'
fmat = read_feature(file)
matshow(fmat)
show()
'''
def write_feature(row_feature, file, period=100000):
    nN, nF = np.shape(row_feature)
    fout = open(file, 'wb')
    fout.write(struct.pack('<i', nN))
    fout.write(struct.pack('<i', period))
    fout.write(struct.pack('<h', nF*4))
    fout.write(struct.pack('<h', 9))
    for i in range(nN):
        for j in range(nF):
            fout.write(struct.pack('<f', row_feature[i, j]))
    fout.close()
'''
from pylab import *
file = r'/home/c2tao/Dropbox/Semester 13/mini/N200108011200-01-01.mfc'
fmat = read_feature(file)
file = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_MFCC/N200108011200-01-01.mfc'
fmat = read_feature(file)
matshow(fmat)
show()
'''

def add_deltas(row_feature):
    row_delta_1 = row_feature[1:]-row_feature[:-1]
    row_delta_2 = row_delta_1[1:]-row_delta_1[:-1]
    #print np.shape(row_delta_1)
    #print np.shape(row_delta_2)
    return np.concatenate((np.concatenate((row_feature[:-2], row_delta_1[:-1]), axis=1),row_delta_2),axis=1)

def mel_filter_out(wav_path):
    from scipy.io import wavfile
    from scikits.talkbox.features import mfcc
    freq, x = wavfile.read(wav_path)
    ceps, mspec, spec = mfcc(x,fs=freq)
    return add_deltas(mspec)

def make_feature(in_folder, out_folder, feature_func=None):
    '''
    in_folder: folder containing only wav files
    out_folder: folder containing only mfc files
    '''
    if not feature_func:
        temp_cfg = open(out_folder + '/temp.cfg', 'w')
        temp_scp = open(out_folder + '/temp.scp', 'w')

        hcopy = """#Coding parameters\nSOURCEFORMAT=WAV\nTARGETKIND=MFCC_Z_E_D_A\nTARGETRATE=100000.0\nSAVECOMPRESSED=F\nSAVEWITHCRC=F\nWINDOWSIZE=320000.0\nUSEHAMMING=T\nPREEMCOEF=0.97\nNUMCHANS=26\nCEPLIFTER=22\nNUMCEPS=12\nENORMALIZE=T\nNATURALREADORDER=TRUE\nNATURALWRITEORDER=TRUE\n"""
        temp_cfg.write(hcopy)
        temp_cfg.close()

        feature_files = []
        for c in sorted(os.listdir(in_folder)):
            temp_scp.write('\"' + in_folder + '/' + c + '\" \"' + out_folder + '/' + c[:-4] + '.mfc' + '\"' + '\n')
            feature_files += '\"' + out_folder + '/' + c[:-4] + '.mfc' + '\"',
        feature_files = sorted(feature_files)
        temp_scp.close()
        # with open (out_folder+'/temp.scp', "r") as myfile:
        # feature_files = myfile.readlines()

        os.system('HCopy -T 1 -C "{}"  -S "{}" '.format(out_folder + '/temp.cfg', out_folder + '/temp.scp'))

        os.remove(out_folder + '/temp.cfg')
        os.remove(out_folder + '/temp.scp')

        return feature_files
    else:
        feature_files = []
        for c in sorted(os.listdir(in_folder)):
            write_feature(feature_func(in_folder +'/'+ c), out_folder + '/' + c[:-4] + '.mfc')
            feature_files += '\"' + out_folder + '/' + c[:-4] + '.mfc' + '\"',
            print 'created feature for: ', c
        feature_files = sorted(feature_files)
        return feature_files
'''
in_folder = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav/'
out_folder = r'/home/c2tao/Dropbox/Semester 12.5/ICASSP 2015 Data/5034 feature/'
print make_feature(in_folder,out_folder)
'''
'''
in_folder = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_mini/'
out_folder = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_mini_MFCC/'
make_feature(in_folder,out_folder,feature_func=mel_filter_out)
'''
def warp(distance_matrix):
    M, N = np.shape(distance_matrix)
    dmap = np.zeros((M + 1, N + 1))
    dmap[1:, 1:] = distance_matrix
    for i in range(1, N + 1):
        dmap[0, i] = float('inf')
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            dmap[i, j] += min(dmap[i - 1, j - 1], dmap[i - 1, j], dmap[i, j - 1])
    return np.min(dmap[:, N]) / N


def cos_dist(A, B):
    return (1.0 - np.dot(A, B.T) / (np.linalg.norm(B, ord=2, axis=1) * np.linalg.norm(A, ord=2, axis=1)[:, None])) / 2


'''
from pylab import *
file = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_MFCC/N200108011200-01-01.mfc'
A = read_feature(file)
B = read_feature(file)[50:100]
file = r'/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_MFCC/N200108011200-01-02.mfc'
C = read_feature(file)[50:100]
print np.shape(A)
print np.shape(B)
print np.shape(C)
print np.shape(cos_dist(A,B))
print 'min distance:',warp(cos_dist(A,B))
print 'min distance:',warp(cos_dist(A,C))
matshow(cos_dist(A,B))
colorbar()
show()
'''


class MLF(object):
    def __init__(self, path, mlf_name=''):
        self.path = path
        self.mlf_name = mlf_name

        lines = open(self.path).readlines()
        wav_list = []
        int_list, tag_list, med_list = [], [], []
        tok_list = []
        log_list = []
        mlf_type = True
        for line in lines[1:]:
            line = line.strip()
            if '"' in line:
                mode = 'wav'
                wav_list += line[3:-5],
                int_temp = []
                tag_temp = []
                med_temp = []
                log_temp = []
                # print wav_list[-1]
            elif line == '.':
                int_list += int_temp,
                tag_list += tag_temp,
                med_list += med_temp,
                log_list += log_temp,
                tok_list = list(set(tok_list + tag_temp))
                continue
            else:
                try:
                    ibeg = int(line.split()[0]) / 100000
                    iend = int(line.split()[1]) / 100000
                    if int_temp and iend == int_temp[-1]: continue
                    int_temp += iend,
                    tag_temp += line.split()[2],
                    med_temp += (ibeg + iend) / 2,
                    log_temp += float(line.split()[3]),
                except:
                    ibeg = 0
                    iend = 0
                    int_temp += iend,
                    tag_temp += line.split()[0],
                    med_temp += (ibeg + iend) / 2,
                    log_temp += 0,
                    mlf_type = False
        self.wav_list = wav_list
        self.int_list = int_list
        self.tag_list = tag_list
        self.med_list = med_list
        self.log_list = log_list
        self.tok_list = sorted(tok_list)
        self.mlf_type = mlf_type

    def fetch(self, med_list):
        return_list = []
        for I, T, Q in zip(self.int_list, self.tag_list, med_list):
            R = []
            pos = 0
            pi = 0
            for i, t, j in zip(I, T, range(len(T))):
                if pos >= len(Q): break
                match = (pi <= Q[pos] and i > Q[pos])
                while match:
                    pos += 1
                    R += j,
                    # R += t,
                    if pos >= len(Q): break
                    match = (pi <= Q[pos] < i)
                pi = i
            return_list += R,
            assert len(R) == len(Q)
        assert len(return_list) == len(med_list)
        return return_list

    def accuracy(self, answer, acc='acc.txt'):
        A = open('phone.txt', 'w')
        for p in answer.tok_list:
            A.write(p + '\n')
        os.system('HResults -e "???" "<s>" -e "???" "</s>" -e "???" sil -e "???" sp -I "{}" "{}" "{}" >> "{}"'.format( \
            answer.path, 'phone.txt', self.path, acc))
        lines = open(acc).readlines()
        return float(lines[-2].split()[1][6:-1])


    def expand(self, phone_mlf, dictionary, edit_file='w2p.led', command='EX\n'):
        file = open(edit_file, 'w')
        file.write(command)
        file.close()
        os.system("""HLEd -l '*' -d "{}" -i "{}" "{}" "{}" """.format( \
            dictionary, phone_mlf, edit_file, self.path))

    def write(self, path='temp.mlf', dot='.rec', selection=()):
        if not selection:
            index_list = range(len(self.tag_list))
            wav_list = self.wav_list
        else:
            index_list, wav_list = zip(*selection)
            # tuples of the form (original_wav_index,new_wav_name)
        if dot == '.lab': self.mlf_type = False
        M = open(path, 'w')
        M.write('#!MLF!#\n')
        for i, w in zip(index_list, wav_list):
            M.write('"*/' + w + dot + '"\n')
            pj = 0
            for j in range(len(self.tag_list[i])):
                if self.mlf_type:
                    w1 = str(pj * 100000)
                    w2 = str(self.int_list[i][j] * 100000)
                    # w1 = str(pj)
                    # w2 = str(self.int_list[i][j])
                    w3 = self.tag_list[i][j]
                    w4 = str(self.log_list[i][j])
                    M.write('{} {} {} {}\n'.format(w1, w2, w3, w4))
                    pj = self.int_list[i][j]
                else:
                    w3 = self.tag_list[i][j]
                    M.write('{}\n'.format(w3))
            M.write('.\n')
        M.close()
        # self.path = path
        return MLF(path)

    def merge(self, ext):
        self.wav_list += ext.wav_list
        self.int_list += ext.int_list
        self.tag_list += ext.tag_list
        self.med_list += ext.med_list
        self.log_list += ext.log_list

        self.tok_list += ext.tok_list
        self.tok_list.sort()


    def wav_tok(self, wav_ind, time_inst):
        # returns the tokens in the list from at
        # when on border, goes to the previos token
        # print self.int_list[wav_ind]
        return self.tag_list[wav_ind][np.nonzero(np.array(self.int_list[wav_ind]) >= time_inst)[0][0]]

    def wav_dur(self, wav_ind, tbeg, tend):
        # returns the tokens in the list from tbeg to tend
        iB = np.nonzero(np.array(self.int_list[wav_ind]) > tbeg)[0][0]
        iE = np.nonzero(np.array(self.int_list[wav_ind]) < tend)[0][-1] + 1 + 1
        # print self.int_list[wav_ind]
        return self.tag_list[wav_ind][iB:iE]


'''
dropbox_path = r'/home/c2tao/Dropbox/'
labels_path = dropbox_path + r'Semester 12.5/ICASSP 2015 Data/'+r'fa.mlf'
M=MLF(labels_path)
print M.wav_tok(0,40)
print M.wav_tok(0,49)
print M.wav_tok(0,52)

print M.wav_tok(0,0)
print M.wav_tok(0,206)

print M.wav_dur(0,40,60)
print M.wav_dur(0,49,60)
print M.wav_dur(0,52,60)

print M.wav_dur(0,40,61)
print M.wav_dur(0,49,61)
print M.wav_dur(0,52,61)

print M.wav_dur(0,40,62)
print M.wav_dur(0,49,62)
print M.wav_dur(0,52,62)


print M.wav_dur(0,50,51)
print M.wav_dur(0,0,206)
'''

'''
A = MLF(r'/home/c2tao/Dropbox/Semester 9.5/40_timit_train_300_11_relabel/result/result.mlf')
print A.wav_list[:10]
print A.int_list[:10]
'''


class Purity(object):
    def __init__(self, pat_MLF, ref_MLF):
        assert (len(pat_MLF.wav_list) == len(ref_MLF.wav_list))
        self.pat_MLF = pat_MLF
        self.ref_MLF = ref_MLF
        # nW number of wav files
        self.nW = len(self.pat_MLF.int_list)
        # feature length of per wav file
        self.nF = [self.pat_MLF.int_list[i][-1] for i in range(self.nW)]
        self.purities = np.zeros(self.nW)
        self.purities_non = np.zeros(self.nW)
        self.purity = 0
        self.purity_non = 0

    def compute(self):
        for j in range(self.nW):
            str1 = []
            ilist = [0] + self.pat_MLF.int_list[j]
            for i in range(len(self.pat_MLF.int_list[j])):
                str1 += [self.pat_MLF.tag_list[j][i]] * (ilist[i + 1] - ilist[i])
            str1 = np.array(str1)

            str2 = []
            ilist = [0] + self.ref_MLF.int_list[j]
            for i in range(len(self.ref_MLF.int_list[j])):
                str2 += [self.ref_MLF.tag_list[j][i]] * (ilist[i + 1] - ilist[i])
            str2 = np.array(str2)

            mat1 = (str1[None, :] == str1[:, None])
            mat2 = (str2[None, :] == str2[:, None])
            mat3 = (mat1 == mat2)
            mat4 = (~mat1 * ~mat2)
            self.purities_non[j] = float(np.sum(mat3) - np.sum(mat4)) / (self.nF[j] * self.nF[j] - np.sum(mat4))
            self.purities[j] = float(np.sum(mat3)) / (self.nF[j] * self.nF[j])
        self.purity = np.mean(self.purities)
        self.purity_non = np.mean(self.purities_non)

        return self.purity


'''
A = MLF(r'/home/c2tao/Dropbox/Semester 9.5/40_timit_train_50_3_relabel/result/result.mlf')
B = MLF(r'/home/c2tao/Dropbox/Semester 9.5/1_timit_train_50_3/result/result.mlf')
C = MLF(r'/home/c2tao/Dropbox/Semester 9.5/timit_baseline/training_transcription7.mlf')

P = Purity(A,C)
P.compute()
print P.purity

Q = Purity(B,C)
Q.compute()
print Q.purity
'''


def average_precision(answer, score):
    I = np.array(sorted(range(len(answer)), key=lambda x: score[x], reverse=True))
    sorted_answer = np.array(map(lambda x: float(answer[I[x]]), range(len(answer))))
    position = np.array(range(len(answer))) + 1
    ap = np.cumsum(sorted_answer) / position
    nz = np.nonzero(sorted_answer)[0]
    return np.mean(ap[nz])


'''
print average_precision([1,1,1,0,0,],[1,1,1,0,-1])
print average_precision([1,1,0,0,1,],[1,1,1,0,0])
'''


def average_precision_minus1(answer, score):
    I = np.array(sorted(range(len(answer)), key=lambda x: score[x], reverse=True))
    sorted_answer = np.array(map(lambda x: float(answer[I[x]]), range(len(answer))))[1:]
    position = np.array(range(len(answer) - 1)) + 1
    ap = np.cumsum(sorted_answer) / position
    nz = np.nonzero(sorted_answer)[0]
    return np.mean(ap[nz])


'''
print average_precision_minus1([1,0,0,],[1,0,-1])
print average_precision_minus1([1,1,0,],[1,0,-1])
print average_precision_minus1([1,1,0,],[1,0,1])
'''
