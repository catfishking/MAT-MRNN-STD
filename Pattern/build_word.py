from zrst import asr
import wave

f = file('bigram.txt','r')

bigram = {}
for line in f:
    print line
    line = line.split()
    bigram[(line[1],line[2])] = line[0]
    print "{}   {}".format(line[1],line[2])

wav_dir = './hey/'
corpus = '/home/troutman/Documents/proj2/cmu/wav_1channel/'
#corpus = '/home/troutman/Documents/proj2/MLDS_HW1_RELEASE_v1/wav/'
target = 'target_s5_c100_iter10/'


label = target + 'result/result.mlf'


A = asr.HTK().readMLF(label)
waveData = dict({})
params = []
for wavefile in A.wav2word:
#    print wavefile
#    print A.wav2word[wavefile]#['sil', 0, 1500000, -500.260468]
#    print A.wav2word[wavefile][-1][1]
    W = wave.open(corpus + wavefile + '.wav')
    scale = float(W.getnframes()) / A.wav2word[wavefile][-1][1]
		
    params = W.getparams()
    for i in range(len(A.wav2word[wavefile])-2):
#	print A.wav2word[wavefile][i]
#	print 'ha2'
#	if(bigram.has_key((A.wav2word[wavefile][i][0],A.wav2word[wavefile][i+2][0]))):
        if(True):
            print A.wav2word[wavefile][i][0]
            print A.wav2word[wavefile][i+2][0]
            print 'ha'
            framechunk = W.readframes(int(scale*(A.wav2word[wavefile][i+2][2]-A.wav2word[wavefile][i][1])))
            word = A.wav2word[wavefile][i][0] + A.wav2word[wavefile][i+2][0]
            if word in waveData:
                try:
                    waveData[word] += framechunk
                except:
                    print word, 'out of memory'
                    pass
            else:
                waveData[word] = framechunk

			
'''
    for word in A.wav2word[wavefile]:
        if(bigram.has_key((word[0])))
            framechunk = W.readframes(int(scale * (word[2] - word[1])))
        if word[0] in waveData:
            try:
                waveData[word[0]] += framechunk
            except:
                print word[0], 'out of memory'
                pass
        else:
            waveData[word[0]] = framechunk'''

for word in waveData:
    print word
    if "<" in word:
        continue
        # this should only happen for words with illegal characters
    S = wave.open(wav_dir + word + '.wav', 'w')
    S.setparams(params)
    S.writeframes(waveData[word])

