import os
def load_feature(in_folder):
    acc = 0
    batch = []
    disk = []
    for c in sorted(os.listdir(in_folder)):
        batch.append(c)
        if acc > 1024*1024*1024:
            acc=0
            batch = []
            disk.append(batch)
        acc += os.path.getsize(in_folder+'/'+c)
    disk.append(batch)
    print len(disk)
    print len(disk[0])
    print disk[0][0]
load_feature('/home/c2tao/Dropbox/Semester 8.5/Corpus_5034wav_MFCC/')
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