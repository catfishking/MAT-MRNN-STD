import os
import shelve
import wave

from zrst.asr import HTK, SYS


EXEroot = '/home/c2tao/Dropbox/ULT_v0.0/'
ASRroot = EXEroot + 'Chinese_ASR/'
am = ASRroot + 'hmmset.mmf'
lm = ASRroot + 'tri.3-gram.lm.ascii'
lex = ASRroot + 'if.lex.ascii'
tied = ASRroot + 'tielist'
config = ASRroot + 'config.cfg'
mapp = ASRroot + 'big5.txt'
# script = 'decodeList.txt'
# script = 'ptvChan.txt'
# result = 'check.mlf'
# result = 'ptvChan.mlf'
#result = 'ptvChan.mlf'

#bigresult = 'ptvLee_big.mlf'
#bigresult = 'ptvChan_big.mlf'
#bigmlf = 'big_label_ptv.mlf'
#ascmlf = 'asc_label_ptv.mlf'
#forced = 'forced.mlf'
#break oov to single characrer
#open(ascmlf)
#Exeroot = ''

def HDecode(script, output_mlf):
    os.system('HDecode -H {} -S {} -t 100.0 100.0 -C {} -i {} -w {} -p 0.0 -s 5.0 {} {}'. \
              format(am, script, config, output_mlf, lm, lex, tied))


def HDecode_English(script, output_mlf):
    ASRroot = EXEroot + 'English_ASR/'
    am = ASRroot + 'hmmdefs'
    lm = ASRroot + 'tri_lm.lc'
    lex = ASRroot + 'dict_nosil'
    tied = ASRroot + 'tiedlist'
    config = ASRroot + 'config.cfg'
    os.system('HDecode -H {} -S {} -t 100.0 100.0 -C {} -i {} -w {} -p 0.0 -s 5.0 {} {}'. \
              format(am, script, config, output_mlf, lm, lex, tied))


def HVite_English(script, output_mlf):
    #mess, dont use yet
    ASRroot = EXEroot + 'English_ASR/'
    am = ASRroot + 'hmmdefs_mono'
    #am      = ASRroot + 'hmmdefs'
    tied = ASRroot + 'phone_list.txt'
    #tied    = ASRroot + 'tiedlist_phone.txt'
    config = ASRroot + 'config.cfg'
    lex = 'temp_asr_lex.txt'
    wdnet = 'temp_asr_mlf.txt'
    grammar = 'temp_asr_gmr.txt'

    tied_list = map(lambda x: x.strip('\n'), open(tied).readlines())
    L = open(lex, 'w')
    for t in tied_list:
        L.write(t + ' ' + t + '\n')
    L.close()
    #pyHTK.HTK().readDCT(lex).writeDCT(grammar,['grammar','sil','sp'])
    HTK().readDCT(lex).writeDCT(grammar, ['full'])
    os.system('HParse {} {}'.format(grammar, wdnet))
    os.system("HVite -D -H {} -S {} -C {} -w {} -l '*' -i {} -p 0.0 -s 0.0 {} {}".format( \
        am,
        script, config,
        wdnet, output_mlf,
        lex, tied
    ))


def HVite_general(am, script, output_mlf, tied):
    #mess, dont use yet
    ASRroot = EXEroot + 'Chinese_ASR/'
    config = ASRroot + 'config.cfg'
    lex = 'temp_asr_lex.txt'
    wdnet = 'temp_asr_mlf.txt'
    grammar = 'temp_asr_gmr.txt'

    tied_list = map(lambda x: x.strip('\n'), open(tied).readlines())
    L = open(lex, 'w')
    for t in tied_list:
        L.write(t + ' ' + t + '\n')
    L.close()
    #pyHTK.HTK().readDCT(lex).writeDCT(grammar,['grammar','sil','sp'])
    HTK().readDCT(lex).writeDCT(grammar, ['full'])
    os.system('HParse {} {}'.format(grammar, wdnet))
    os.system("HVite -D -H '{}' -S {} -C {} -w {} -l '*' -i {} -p 0.0 -s 0.0 {} {}".format( \
        am,
        script, config,
        wdnet, output_mlf,
        lex, tied
    ))


def HVite(input_mlf, script, output_mlf):
    os.system("HVite -a -I '{}' -D -H '{}' -S '{}' -C '{}' -l '*' -i '{}' -p 0.0 -s 0.0 '{}' '{}'". \
              format(input_mlf, am, script, config, output_mlf, lex, tied))


def HVite_phone(input_mlf, script, output_mlf):
    os.system("HVite -a -m -I '{}' -D -H '{}' -S '{}' -C '{}' -l '*' -i '{}' -p 0.0 -s 0.0 '{}' '{}'". \
              format(input_mlf, am, script, config, output_mlf, lex, tied))


def HVite_state(input_mlf, script, output_mlf):
    os.system("HVite -a -f -I {} -D -H {} -S {} -C {} -l '*' -i {} -p 0.0 -s 0.0 {} {}". \
              format(input_mlf, am, script, config, output_mlf, lex, tied))


def HVite_nbest(script, output_mlf):
    #note: not complete, still needs wordnet
    os.system("HVite -n 4 20 -D -H {} -S {} -C {} -l '*' -i {} -p 0.0 -s 0.0 {} {}". \
              format(am, script, config, output_mlf, lex, tied))


def removeOOV(oov, asc):
    O = open(oov).readlines()
    L = open(lex).readlines()
    A = open(asc, 'w')
    zhumap = dict({})
    for line in L:
        line = line.strip('\n')
        p = line.split('\t')
        if 'sil' in line or not line: continue
        zhumap[p[0]] = p[1]

    for line in O:
        phrase = ''
        for i in range(len(line)):
            if line[i] != '[': continue
            character = line[i:i + 6]
            phrase += character

        if not phrase or phrase in zhumap:
            A.write(line)
            continue

        print phrase
        for i in range(len(line)):
            if line[i] != '[': continue
            character = line[i:i + 6]
            A.write(character + '\n')


def char2homo(input, output, bracket=True):
    x = shelve.open('../../Semester 8.5/_readonly/char2homo.dat')
    lexicon = x['lexicon']
    psymbol = x['psymbol']
    #C = open('../5034_asc.txt').readlines()
    D = open(input).readlines()
    E = open(output, 'w')
    for line in D:
        temp = line
        while '[' in temp:
            char = temp[temp.index('['):temp.index('[') + 6]
            #print char
            homo = psymbol[lexicon[char]]
            if bracket:
                temp = temp.replace(char, '(' + homo + ')')
            else:
                temp = temp.replace(char, homo)
                #print char,homo
        E.write(temp)
    x.close()


def asc2big(asc, big):
    D = open(mapp).readlines()
    bigmap = dict({})
    for line in D:
        p = line.strip('\n').split()
        bigmap[p[0]] = p[1]
        #try:  bigmap[p[0]] = p[1]
        #except: continue
    I = open(asc).readlines()
    O = open(big, 'w')
    for line in I:
        while '[' in line and ']' in line:
            #character = line[line.index('['):line.index(']')+1]
            try:
                character = line[line.index('['):line.index('[') + 6]
                line = line.replace(character, bigmap[character])
            except:
                character = line[line.index(']') - 5:line.index(']') + 1]
                line = line.replace(character, bigmap[character])
        O.write(line)


def big2asc(big, asc):
    D = open(mapp).readlines()
    bigmap = dict({})
    for line in D:
        p = line.strip('\n').split()
        try:
            bigmap[p[1]] = p[0]
        except:
            continue
    I = open(big).readlines()
    O = open(asc, 'w')
    for line in I:
        while True:
            if all(ord(c) < 128 for c in line):  break
            for c in line:
                try:
                    c.decode('ascii')
                except:
                    character = line[line.index(c):line.index(c) + 2]
                    line = line.replace(character, bigmap[character], 1)
                    break
        O.write(line)


def word2phone(wordmlf, phonemlf):
    text = open(wordmlf).readlines()
    B = open(phonemlf, 'w')
    for line in text:
        if len(line.split()) == 1:
            B.write(line)
        elif 's' in line:
            B.write(line)
        else:
            #print line
            word = line.strip('\n').split()
            p_list = word[2].split('p')[1:]
            n = len(p_list)
            beg = int(word[0])
            end = int(word[1])
            interval = (end - beg) / n
            likelihood = float(word[3]) / n
            for i in range(n):
                B.write(str(beg + i * interval))
                B.write(' ')
                B.write(str(beg + (i + 1) * interval))
                B.write(' ')
                B.write('p' + p_list[i])
                B.write(' ')
                B.write(str(likelihood))
                B.write('\n')


def lab2wav(label, wav_dir, corpus):
    '''
    input: transcription label
    output: wav folder
    option: wav corpus
    '''
    SYS().cldir(wav_dir)
    A = HTK().readMLF(label)
    waveData = dict({})
    params = []
    # print 'ready'
    for wavefile in A.wav2word:
        W = wave.open(corpus + wavefile + '.wav')
        scale = float(W.getnframes()) / A.wav2word[wavefile][-1][1]
        params = W.getparams()
        for word in A.wav2word[wavefile]:
            framechunk = W.readframes(int(scale * (word[2] - word[1])))
            if word[0] in waveData:
                try:
                    waveData[word[0]] += framechunk
                except:
                    print word[0], 'out of memory'
                    pass
            else:
                waveData[word[0]] = framechunk
    for word in waveData:
        if "<" in word:
            continue
            # this should only happen for words with illegal characters
        S = wave.open(wav_dir + word + '.wav', 'w')
        S.setparams(params)
        S.writeframes(waveData[word])