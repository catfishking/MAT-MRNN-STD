import os
import shutil
import time
import wave
import util
from configs import *


class SYS(object):
    def __init__(self):
        pass

    def mkdir(self, path):
        try:
            print "created" + path
            os.mkdir(path)
        except:
            pass
        return self

    def rmdir(self, path):
        try:
            print "removed" + path
            shutil.rmtree(path)
        except:
            pass
        return self

    def cldir(self, path):
        self.rmdir(path)
        self.mkdir(path)
        return self


class HTK(object):
    def __init__(self):
        # MLF
        self.wav2word = dict({})
        self.word2wav = dict({})
        self.wav_list = []
        # DCT
        self.word2pone = dict({})
        self.pone2word = dict({})
        self.word_list = []
        # SCP
        self.directory = None

    def readSCP(self, path):
        self.directory = path
        return self

    def writeSCP(self, path, opt_list=(), mfcpath=''):
        scp = open(path, 'w')
        contents = os.listdir(self.directory)
        for file in contents:
            if 'hcopy' in opt_list:
                scp.write('"' + self.directory + file + '" ')
            scp.write('"' + mfcpath + file[:-3] + 'mfc' + '"' + '\n')
        return self

    def convertMLF2DCT(self, threshold=2):
        "word is deleted if count in histogram is smaller than threshold, threshold [0, +inf]"
        # "word is deleted if percentage in histogram is smaller than threshold, threshold (0,1)"
        # note: should run "testing" right after this if threshold is not 0

        word2pone = self.word2pone
        pone2word = self.pone2word
        word_list = self.word_list
        self.word2pone = dict({})
        self.pone2word = dict({})
        self.word_list = []

        count = dict({})
        total = 0
        for word in self.word2wav:
            count[word] = len(self.word2wav[word])
            total += count[word]

        for word in self.word2wav:
            # if float(count[word])/total >= threshold:
            if count[word] >= threshold:
                self.word_list.append(word)
                self.word2pone[word] = word2pone[word]
                pone = word2pone[word]
                self.pone2word[pone] = word
        return self

    def writeDCT(self, path, opt_list=()):
        dct = open(path, 'w')

        if 'model' in opt_list:
            phones = set()
            for word in self.word_list:
                for phone in self.word2pone[word]:
                    phones.add(phone)
            phones = sorted(list(phones))
            for phone in phones:
                dct.write(phone + '\n')
            for extra in opt_list:
                if extra != 'model':
                    dct.write(extra + '\n')
            return self

        if 'grammar' in opt_list:
            dct.write('$word=')
            for word in self.word_list:
                dct.write(word)
                if self.word_list.index(word) != len(self.word_list) - 1: dct.write(' | ')
            dct.write(' ;\n')
            if 'sp' in opt_list:
                dct.write('$word_sp=$word sp;\n')
                dct.write('(sil {$word_sp} $word sil)\n')
            else:
                dct.write('(sil <$word> sil)\n')
            return self

        if 'full' in opt_list:
            dct.write('$word=')
            for word in self.word_list:
                dct.write(word)
                if self.word_list.index(word) != len(self.word_list) - 1: dct.write(' | ')
            dct.write(' ;\n')
            dct.write('(<$word>)\n')
            return self

        sortedword_list = []
        for word in self.word_list:
            sortedword_list.append(word)
        sortedword_list.sort()
        for word in sortedword_list:
            phone = self.word2pone[word]
            pstring = ""
            for p in range(len(phone)):
                pstring += phone[p]
                if p != len(phone) - 1:
                    pstring += ' '
            dct.write(word + ' ' + pstring + '\n')
        for extra in opt_list:
            dct.write(extra + ' ' + extra + '\n')
        return self

    def readDCT(self, path, opt_list=()):
        text = [t.rstrip('\n').split() for t in open(path).readlines()]
        self.word2pone = ({})
        self.pone2word = ({})
        self.word_list = []
        for t in text:
            if t[0] in ['sil', 'sp']:  continue
            pone = tuple(t[1:])
            word = t[0]
            self.word_list.append(word)
            self.word2pone[word] = pone
            self.pone2word[pone] = word
        return self

    @staticmethod
    def readMLF_setmode(path):
        if path[-4:] == '.txt':
            return 'dump'
        elif path[-4:] == '.mlf':
            text = [t.rstrip('\n').split() for t in open(path).readlines()]
            if len(text[2]) > 1:
                return 'time'
            if len(text[2]) == 1:
                return 'word'

    @staticmethod
    def readMLF_filter(text, mode, filter_list):
        pop_list = []
        for i in range(len(text)):
            # print text[i]
            if not text[i]:
                pop_list.append(i)
                continue
            if text[i][0][0] == '.' or text[i][0][0] == '#':
                pop_list.append(i)
                continue

            for banned in filter_list:
                if mode == 'word' and '.' not in text[i][0]:
                    if text[i][0] == banned:
                        pop_list.append(i)
                        continue
                if mode == 'time' and '.' not in text[i][0]:
                    if text[i][2] == banned:
                        pop_list.append(i)
                        continue
        pop_list.reverse()
        for i in pop_list:
            text.pop(i)
        return text

    def readMLF(self, path, opt_list=()):
        "reads in MLF file, set words to filter in opt_list"
        mode = self.readMLF_setmode(path)
        raw_text = [t.rstrip('\n').split() for t in open(path).readlines()]
        text = self.readMLF_filter(raw_text, mode, opt_list)

        for t in text:
            if '.' in t[0]:
                if mode == 'time':
                    wav = t[0][3:-5]
                if mode == 'word':
                    wav = t[0][3:-5]
                if mode == 'dump':
                    wav = t[0][:-4]
                self.wav_list.append(wav)
            else:
                if mode == 'time':
                    word = t[2]
                    try:
                        contentword = [word, int(t[0]), int(t[1]), float(t[3])]
                        contentwav = [wav, int(t[0]), int(t[1]), float(t[3])]
                    except:
                        contentword = [word, int(t[0]), int(t[1])]
                        contentwav = [wav, int(t[0]), int(t[1])]
                if mode == 'word':
                    word = t[0]
                    contentword = [word, ]
                    contentwav = [wav, ]
                if mode == 'dump':
                    # pones = tuple(map(lambda x: 'phone' + x, t))
                    pones = tuple(map(lambda x: 'p' + x, t))
                    if pones not in self.pone2word:
                        # wordID = 'word' + str(len(self.word2pone) + 1)
                        wordID = ''
                        for phone in pones:
                            wordID += phone
                        self.pone2word[pones] = wordID
                        self.word2pone[wordID] = pones
                        self.word_list.append(wordID)
                    word = self.pone2word[pones]
                    contentword = [word, ]
                    contentwav = [wav, ]
                # if word not in opt_list:
                try:
                    self.wav2word[wav] += contentword,
                except:
                    self.wav2word[wav] = contentword,
                try:
                    self.word2wav[word] += contentwav,
                except:
                    self.word2wav[word] = contentwav,
        self.word_list = sorted(self.word_list)
        return self

    def writeMLF(self, path, opt_list=(), dot3='.lab'):
        mlf = open(path, 'w')
        if 'lm' in opt_list:
            for wav in self.wav_list:
                mlf.write('<s> ')
                for i in range(len(self.wav2word[wav])):
                    if 'phone' in opt_list:
                        word = self.wav2word[wav][i][0]
                        phones = self.word2pone[word]
                        for j in range(len(phones)):
                            mlf.write(phones[j] + ' ')
                    else:
                        mlf.write(self.wav2word[wav][i][0] + ' ')
                mlf.write('</s>\n')
            return self

        mlf.write('#!MLF!#\n')
        for wav in self.wav_list:
            mlf.write('"*/' + wav + dot3 + '"\n')
            if 'sil' in opt_list: mlf.write('sil\n')
            for i in range(len(self.wav2word[wav])):
                if 'phone' in opt_list:
                    word = self.wav2word[wav][i][0]
                    phones = self.word2pone[word]
                    for j in range(len(phones)):
                        mlf.write(phones[j] + '\n')
                        # if 'sp' in opt_list and j!= len(phones)-1: mlf.write('sp\n')
                else:
                    mlf.write(self.wav2word[wav][i][0] + '\n')

                if 'sp' in opt_list and i != len(self.wav2word[wav]) - 1: mlf.write('sp\n')
            if 'sil' in opt_list: mlf.write('sil\n')
            mlf.write('.\n')
        return self

    def cross(self, result):
        composition = dict({})
        for word in self.word2wav:
            for wav in self.word2wav[word]:
                abeg = wav[1]
                aend = wav[2]
                adur = aend - abeg
                for seg in result.wav2word[wav[0]]:
                    rdur = 0
                    rbeg = seg[1]
                    rend = seg[2]
                    if rbeg <= abeg and rend >= abeg:
                        rdur = rend - abeg
                    if rbeg >= abeg and rend <= aend:
                        rdur = rend - rbeg
                    if rbeg <= aend and rend >= aend:
                        rdur = aend - rbeg
                    if rbeg <= abeg and rend >= aend:
                        rdur = aend - abeg
                    try:
                        composition[(word, seg[0])] += rdur
                    except:
                        composition[(word, seg[0])] = 0
        return composition

    def seg_cross(self, result):
        # note: we are A, looking for overlap with R(eference)
        seg_composition = dict({})
        for word in self.word2wav:
            for wav in self.word2wav[word]:
                abeg = wav[1]
                aend = wav[2]
                adur = wav[2] - wav[1]
                if word.count('p') != 0:
                    phone_num = word.count('p')
                else:
                    phone_num = word.count('[')
                pattern = []
                for seg in result.wav2word[wav[0]]:
                    rbeg = seg[1]
                    rend = seg[2]
                    phone = seg[0]
                    thresh = 0.5 / phone_num
                    if (rbeg < abeg and rend > abeg and float(rend - abeg) / adur > thresh) \
                            or (rbeg >= abeg and rend <= aend and float(rend - rbeg) / adur > thresh) \
                            or (rbeg < aend and rend > aend and float(aend - rbeg) / adur > thresh) \
                            or (rbeg <= abeg and rend >= aend and float(aend - abeg) / adur > thresh):
                        pattern.append(phone)
                pattern = tuple(pattern)
                if word in seg_composition:
                    if pattern in seg_composition[word]:
                        seg_composition[word][pattern] += 1
                    else:
                        seg_composition[word][pattern] = 1
                else:
                    seg_composition[word] = dict({})
                    seg_composition[word][pattern] = 1
        return seg_composition


    def percentage(self, mlf):
        composition = self.cross(mlf)
        percent = dict({})
        for rword in self.word2wav:
            total = 0
            percent[rword] = dict({})
            for aword in mlf.word2wav:
                try:
                    if composition[(rword, aword)] != 0:
                        total += composition[(rword, aword)]
                        percent[rword][aword] = composition[(rword, aword)]
                except:
                    pass
            percent[rword] = self.normalize_dict(percent[rword])
        return percent

    def renameMLF2MLF(self, mlf):
        percent = self.percentage(mlf)
        name_change = dict({})
        for rword in percent:
            maxvalue = 0
            for aword in percent[rword]:
                try:
                    if percent[rword][aword] >= maxvalue:
                        name_change[rword] = aword
                        maxvalue = percent[rword][aword]
                except:
                    pass
        # pprint(percent)
        # pprint(name_change)

        word2pone = self.word2pone
        pone2word = self.pone2word
        word_list = self.word_list
        word2wav = self.word2wav

        self.word2pone = dict({})
        self.pone2word = dict({})
        self.word_list = []
        self.word2wav = dict({})

        for word in word2wav:
            try:
                cword = name_change[word]
            except:
                # note: this case should only happen in the first pass
                # this means that a word is in the decoded results with 0 duration
                name_change[word] = word
                cword = name_change[word]
            # print cword,word
            try:
                self.word2pone[cword] = word2pone[word]
            except:
                word2pone[word] = word,
                self.word2pone[cword] = word2pone[word]
            self.pone2word[word2pone[word]] = cword
            self.word_list.append(cword)
            self.word2wav[cword] = word2wav[word]
        # pprint(self.wav2word)
        for wav in self.wav_list:
            for word in self.wav2word[wav]:
                word[0] = name_change[word[0]]

    def normalize_dict(self, dictionary):
        total = 0
        normalized = dict({})
        for word in dictionary:
            total += dictionary[word]
        for word in dictionary:
            normalized[word] = float(dictionary[word]) / total
        return normalized


class ASR(object):
    def __init__(self, corpus='', target='./', label=(), dump='IDump.txt', nState=3, nFeature=39, feature_func=None,
                 do_copy=False, pad_silence=True):
        self.do_copy = do_copy
        self.corpus = corpus
        # self.label = label
        self.target = target
        SYS().mkdir(target)
        self.X = dict({})
        self.setname()
        self.time = 0
        self.offset = 0
        self.dump = dump
        self.nFeature = nFeature
        self.nState = nState

        self.pad_silence = pad_silence

        if self.pad_silence:
            self.sil_tag = 'grammar'
        else:
            self.sil_tag = 'full'
        if feature_func:
            self.user_feature = True
            self.feature_func = feature_func
        else:
            self.user_feature = False
        #os.chdir(self.target)
        #os.chdir('..')

    def slice(self, corpus=(), target=(), label=()):
        if not corpus: corpus = self.X['corpus_dir']
        if not label:  label = self.X['result_mlf']
        if not target: target = self.X['acoust_dir']
        SYS().cldir(target)
        A = HTK().readMLF(label)
        waveData = dict({})
        params = []
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
            S = wave.open(target + word + '.wav', 'w')
            S.setparams(params)
            S.writeframes(waveData[word])

    def readASR(self, target):
        # shutil.rmtree(self.target)
        try:
            shutil.rmtree(self.target)
        except:
            pass
        shutil.copytree(target, self.target)

    def writeASR(self, target):
        try:
            shutil.rmtree(target)
        except:
            pass
        shutil.copytree(self.target, target)

    def record_comment(self, tag):
        open(self.X['evalua_txt'], 'a').write(tag + '\n')

    def record_time(self):
        open(self.X['evalua_txt'], 'a').write('total_time ' + str(time.time() - self.time) + '\n')

    def consistency_check(self):
        pass

    def clean(self):
        SYS() \
            .cldir(self.target) \
            .cldir(self.X['markov_dir']) \
            .cldir(self.X['answer_dir']) \
            .cldir(self.X['libray_dir']) \
            .cldir(self.X['result_dir']) \
            .cldir(self.X['lanmdl_dir']) \
            .cldir(self.X['acoust_dir'])

        # initiation
        open(self.X['hcopie_cfg'], 'w').write(hcopy)
        if self.user_feature:
            open(self.X['config_cfg'], 'w').write(config_user)
        else:
            open(self.X['config_cfg'], 'w').write(config)
        # open(self.X['protos_txt'], 'w').write(proto)
        self.build_proto()

        # training
        open(self.X['addsil_scp'], 'w').write(addsil)
        open(self.X['addmix_scp'], 'w').write(addmix)
        #addmix_all = """MU 3 \{sil.state[2-4].mix}\nMU +1 {*.state[2-%d].mix}\n""" % self.nState
        #open(self.X['addall_scp'], 'w').write(addmix_all)

        # reference
        # if self.label:
        # HTK().readMLF(self.X['refere_mlf'], ['sp', 'sil']).writeMLF(self.X['reflab_mlf'], dot3='.lab')

    def build_proto(self):
        a_state = self.nState
        a_feature = self.nFeature
        state = str(a_state + 2)
        A = open(self.X['protos_txt'], 'w')
        if not self.user_feature:
            A.write('~o <VECSIZE> {} <MFCC_Z_E_D_A>\n'.format(str(a_feature)))
        else:
            A.write('~o <VECSIZE> {} <USER>\n'.format(str(a_feature)))
        A.write('~h "proto"\n')
        A.write('<BeginHMM>\n')
        A.write('<NumStates> ' + state + '\n')
        for i in range(2, a_state + 2):
            A.write('<State> ' + str(i) + '\n')

            A.write('<Mean> ' + str(a_feature) + '\n')
            for ___ in range(a_feature): A.write('0.0 ')
            A.write('\n')

            A.write('<Variance> ' + str(a_feature) + '\n')
            for ___ in range(a_feature): A.write('1.0 ')
            A.write('\n')

        A.write('<TransP> ' + state + '\n')
        for i in range(a_state + 2):
            for j in range(a_state + 2):
                if i == 0 and j == 1:
                    A.write('1.0')
                elif i == 0 and j == 0:
                    A.write('0.0')
                elif i == a_state + 1:
                    A.write('0.0')
                elif i + 1 == j or i == j:
                    A.write('0.5')
                else:
                    A.write('0.0')
                A.write(' ')
            A.write('\n')
        A.write('<EndHMM>\n')
        A.close()

    def setname(self):
        self.X['corpus_dir'] = self.corpus
        self.X['featur_dir'] = self.X['corpus_dir'][:-1] + '_MFCC/'

        self.X['markov_dir'] = self.target + 'hmm/'
        self.X['answer_dir'] = self.target + 'answer/'
        self.X['libray_dir'] = self.target + 'library/'
        self.X['result_dir'] = self.target + 'result/'
        self.X['lanmdl_dir'] = self.target + 'lm/'
        self.X['acoust_dir'] = self.target + 'ac/'

        # initiation
        self.X['hcopie_cfg'] = self.X['libray_dir'] + 'hcopy.cfg'
        self.X['config_cfg'] = self.X['libray_dir'] + 'config.cfg'
        self.X['protos_txt'] = self.X['libray_dir'] + 'proto'

        # feature
        self.X['wavlst_scp'] = self.X['libray_dir'] + 'list.scp'
        self.X['wavhcp_scp'] = self.X['libray_dir'] + 'hcopy.scp'

        # training
        self.X['hmmdef_hmm'] = self.X['markov_dir'] + 'hmmdef'
        self.X['macros_hmm'] = self.X['markov_dir'] + 'macros'
        self.X['models_hmm'] = self.X['markov_dir'] + 'models'
        self.X['addsil_scp'] = self.X['libray_dir'] + 'sil1.hed'
        self.X['addmix_scp'] = self.X['libray_dir'] + 'mix2_10.hed'
        self.X['addall_scp'] = self.X['libray_dir'] + 'mixall.hed'

        # testing
        self.X['result_mlf'] = self.X['result_dir'] + 'result.mlf'
        self.X['accrcy_txt'] = self.X['result_dir'] + 'accuracy.txt'
        self.X['evalua_txt'] = self.X['result_dir'] + 'evaluation.txt'

        # forward
        self.X['answer_mlf'] = self.X['answer_dir'] + 'answer_word.mlf'
        self.X['phonei_mlf'] = self.X['answer_dir'] + 'answer_phone.mlf'
        self.X['phonep_mlf'] = self.X['answer_dir'] + 'answer_phone_sp.mlf'
        self.X['dictry_dct'] = self.X['libray_dir'] + 'dictionary.txt'
        self.X['modeli_dct'] = self.X['libray_dir'] + 'phones.txt'
        self.X['modelp_dct'] = self.X['libray_dir'] + 'phones_sp.txt'
        self.X['grmmri_dct'] = self.X['libray_dir'] + 'grammar.txt'
        self.X['grmmrp_dct'] = self.X['libray_dir'] + 'grammar_sp.txt'
        self.X['wdneti_dct'] = self.X['libray_dir'] + 'wordnet.txt'
        self.X['wdnetp_dct'] = self.X['libray_dir'] + 'wordnet_sp.txt'

        # reference
        # self.X['refere_mlf'] = self.label
        self.X['transf_mlf'] = self.X['result_dir'] + 'transcription.mlf'
        self.X['reflab_mlf'] = self.X['result_dir'] + 'clean_reference.mlf'
        self.X['consis_txt'] = self.X['result_dir'] + 'ASR_accuracy.txt'

        # language
        self.X['inlang_mlf'] = self.X['lanmdl_dir'] + 'sentences.txt'
        self.X['wrdcnt_txt'] = self.X['lanmdl_dir'] + 'wordcount.txt'
        self.X['outlan_txt'] = self.X['lanmdl_dir'] + 'lm.txt'
        self.X['lanres_mlf'] = self.X['lanmdl_dir'] + 'larvevocab_result.mlf'
        self.X['landct_dct'] = self.X['lanmdl_dir'] + 'dictionary_large.txt'
        self.X['tielst_txt'] = self.X['lanmdl_dir'] + 'tielist.txt'

        self.X['biwnet_txt'] = self.X['lanmdl_dir'] + 'biwnet.txt'
        self.X['bigram_txt'] = self.X['lanmdl_dir'] + 'bigram.txt'
        self.X['wdlist_txt'] = self.X['lanmdl_dir'] + 'wdlist.txt'
        self.X['bufdct_dct'] = self.X['lanmdl_dir'] + 'temp_dictionary_large.txt'
        # pattern
        self.X['flattn_mlf'] = self.X['lanmdl_dir'] + 'flat_mlf.mlf'
        self.X['flattn_dct'] = self.X['lanmdl_dir'] + 'flat_dictionary.txt'

    def feature(self):

        HTK() \
            .readSCP(self.X['corpus_dir']) \
            .writeSCP(self.X['wavlst_scp'], [], self.X['featur_dir']) \
            .writeSCP(self.X['wavhcp_scp'], ['hcopy'], self.X['featur_dir'])

        try:    os.mkdir(self.X['featur_dir'])
        except: return

        SYS().cldir(self.X['featur_dir'])
        if not self.user_feature:
            os.system('HCopy -T 1 -C "{}"  -S "{}" '.format(
                self.X['hcopie_cfg'], self.X['wavhcp_scp']))
        else:
            util.make_feature(self.X['corpus_dir'],self.X['featur_dir'],self.feature_func)

    def exp_feature(self, wav_directory, wav_list):
        script = 'temp_hcopy.txt'
        feature_directory = wav_directory[:-1] + '_MFCC/'

        HTK() \
            .readSCP(wav_directory) \
            .writeSCP(wav_list, [], feature_directory) \
            .writeSCP(script, ['hcopy'], feature_directory)

        try:
            os.mkdir(feature_directory)
        except:
            return

        os.system('HCopy -T 1 -C "{}"  -S "{}" '.format(
            self.X['hcopie_cfg'], script))

    def training_macros(self):
        macros = open(self.X['macros_hmm'], 'w')
        if self.user_feature:
            macros.write(macrosf_user.replace('X',str(self.nFeature)))
        else:
            macros.write(macrosf)
        map(lambda x: macros.write(x), open(self.X['markov_dir'] + 'vFloors').readlines())

    def training_hmmdef(self):
        models = open(self.X['models_hmm'], 'w')
        text = open(self.X['hmmdef_hmm']).readlines()
        model_list = [t.rstrip('\n') for t in open(self.X['modeli_dct']).readlines()]
        for model in model_list:
            if model == 'sil':
                self.training_silence(models)
                return
            models.write('~h "' + model + '"\n')
            transcribe = False
            for line in text:
                if transcribe:  models.write(line)
                if line == '~h "hmmdef"\n': transcribe = True

    def training_silence(self, file):
        file.write('~h \"sil\"\n')
        file.write('<BEGINHMM>\n<NUMSTATES> 5\n')
        text = [t.rstrip('\n') for t in open(self.X['hmmdef_hmm']).readlines()]

        state_pos = [i for i in range(len(text)) if 'STATE' in text[i]]
        for j in [2, 3, 4]:
            file.write('<STATE> {}'.format(str(j)))
            for i in range(state_pos[1] + 1, state_pos[2]):
                file.write(text[i])
        file.write(siltransition)

    def training_shortpause(self):
        models = open(self.X['models_hmm'], 'a')
        text = open(self.X['models_hmm']).readlines()
        models.write('~h "sp"\n')
        models.write('<BEGINHMM>\n<NUMSTATES> 3\n<STATE> 2\n')
        state_pos = [i for i in range(len(text)) if 'STATE' in text[i]]
        for i in range(state_pos[1] + 1, state_pos[2]):
            models.write(text[i])
        models.write(sptransition)

    def training(self, mode='default'):
        if mode == 'default':
            iter = [10, 10, 10, 1]
            t_script = self.X['addmix_scp']
            # initiation
            os.system('HCompV -T 2 -D -C "{}"  -o "{}"  -f 0.01 -m -S "{}"  -M "{}"  "{}" '.format( \
                self.X['config_cfg'], 'hmmdef',
                self.X['wavlst_scp'], self.X['markov_dir'],
                self.X['protos_txt']
            ))
            self.training_macros()
            self.training_hmmdef()

            for i in range(iter[0]):
                os.system(
                    'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format( \
                        self.X['config_cfg'], self.X['phonei_mlf'],
                        self.X['wavlst_scp'], self.X['macros_hmm'],
                        self.X['models_hmm'], self.X['markov_dir'],
                        self.X['modeli_dct']
                    ))
            # add short pause
            self.training_shortpause()
            os.system('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format( \
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['markov_dir'], self.X['addsil_scp'],
                self.X['modelp_dct']
            ))
            for i in range(iter[1]):
                os.system(
                    'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format( \
                        self.X['config_cfg'], self.X['phonep_mlf'],
                        self.X['wavlst_scp'], self.X['macros_hmm'],
                        self.X['models_hmm'], self.X['markov_dir'],
                        self.X['modelp_dct']
                    ))
            # increase mixture
            for i in range(iter[3]):
                os.system('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
                    self.X['macros_hmm'], self.X['models_hmm'],
                    self.X['markov_dir'], t_script,
                    self.X['modelp_dct']
                ))
                for ___ in range(iter[2]):
                    os.system(
                        'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                            self.X['config_cfg'], self.X['phonep_mlf'],
                            self.X['wavlst_scp'], self.X['macros_hmm'],
                            self.X['models_hmm'], self.X['markov_dir'],
                            self.X['modelp_dct']
                        ))
        elif mode == 'addmix':
            # addmix_all = """MU +1 {*.state[2-%d].mix}\n"""%(self.nState) ### big mistake
            addmix_all = """MU +1 {*.state[2-%d].mix}\n""" % (self.nState + 1)
            open(self.X['addall_scp'], 'w').write(addmix_all)
            t_script = self.X['addall_scp']
            os.system('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['markov_dir'], t_script,
                self.X['modelp_dct']
            ))
            for i in range(10):
                os.system(
                    'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                        self.X['config_cfg'], self.X['phonep_mlf'],
                        self.X['wavlst_scp'], self.X['macros_hmm'],
                        self.X['models_hmm'], self.X['markov_dir'],
                        self.X['modelp_dct']
                    ))
        elif mode == 'keepmix':
            for i in range(10):
                os.system(
                    'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                        self.X['config_cfg'], self.X['phonep_mlf'],
                        self.X['wavlst_scp'], self.X['macros_hmm'],
                        self.X['models_hmm'], self.X['markov_dir'],
                        self.X['modelp_dct']
                    ))

    def exp_training(self, iter=(10, 10, 10)):
        # initiation
        os.system('HCompV -T 2 -D -C "{}"  -o "{}"  -f 0.01 -m -S "{}"  -M "{}"  "{}" '.format(
            self.X['config_cfg'], 'hmmdef',
            self.X['wavlst_scp'], self.X['markov_dir'],
            self.X['protos_txt']
        ))
        self.training_macros()
        self.training_hmmdef()

        for i in range(iter[0]):
            os.system(
                'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                    self.X['config_cfg'], self.X['phonei_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modeli_dct']
                ))
        # add short pause
        self.training_shortpause()
        os.system('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
            self.X['macros_hmm'], self.X['models_hmm'],
            self.X['markov_dir'], self.X['addsil_scp'],
            self.X['modelp_dct']
        ))
        for i in range(iter[1]):
            os.system(
                'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                    self.X['config_cfg'], self.X['phonep_mlf'],
                    self.X['wavlst_scp'], self.X['macros_hmm'],
                    self.X['models_hmm'], self.X['markov_dir'],
                    self.X['modelp_dct']
                ))
        # increase mixture
        for i in range(8):
            os.system('HHEd -T 2 -H "{}"  -H "{}"  -M "{}"  "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['markov_dir'], self.X['addmix_scp'],
                self.X['modelp_dct']
            ))
            for ___ in range(iter[2]):
                os.system(
                    'HERest -C "{}"  -I "{}"  -t 250.0 150.0 1000.0 -S "{}"  -H "{}"  -H "{}"  -M "{}"  "{}" '.format(
                        self.X['config_cfg'], self.X['phonep_mlf'],
                        self.X['wavlst_scp'], self.X['macros_hmm'],
                        self.X['models_hmm'], self.X['markov_dir'],
                        self.X['modelp_dct']
                    ))

    def replace_string(self, path, old, new):
        text = open(path).readlines()
        W = open(path, 'w')
        for line in text:
            line = line.replace(old, new)
            W.write(line)

    def report(self, t0):
        R = HTK()
        R.readMLF(self.X['result_mlf'], ['sp', 'sil', '<s>', '</s>'])
        R.readDCT(self.X['dictry_dct'], ['sp', 'sil', '<s>', '</s>'])
        W = open(self.X['evalua_txt'], 'a')
        W.write('word_count ' + str(len(R.word2wav)) + '\n')
        # W.write('word_entropy ' + str(R.entropy()) + '\n')
        '''
        if self.label:
            A = HTK().readMLF(self.X['refere_mlf'], ['sp', 'sil', '<s>', '</s>'])
            R.renameMLF2MLF(A)
            R.writeMLF(self.X['transf_mlf'], dot3='.rec')
            #-e "???" <s> -e "???" </s>
            os.system('HResults -e "???" sil -e "???" sp  -I "{}"  "{}"  "{}"  >> "{}" '.format (\
                self.X['reflab_mlf'], self.X['modelp_dct'], 
                self.X['transf_mlf'], self.X['consis_txt']
            ))
            W.write('conditional_entropy ' + str(R.conditional_entropy(A))+'\n')
            W.write('joint_entropy ' + str(R.joint_entropy(A))+'\n')
        '''
        # ms windows: t is wall seconds elapsed (floating point)
        # linux: t is CPU seconds elapsed (floating point)

        t = time.time() - t0
        W.write('decode_time ' + str(t) + '\n')


    def bigram2wordnet(self):
        A = open(self.X['wdlist_txt'], 'w')
        w2w = HTK().readMLF(self.X['answer_mlf']).word2wav
        for word in w2w:
            A.write(word + '\n')
        A.write('</s>\n<s>\n')
        A.close()
        os.system('sort ' + self.X['wdlist_txt'])
        self.lm_dict()

        os.system('HLStats -s "<s>" "</s>" -b "{}"  -o "{}"  "{}" '.format(
            self.X['bigram_txt'], self.X['wdlist_txt'], self.X['answer_mlf']
        ))

        os.system('HBuild -s "<s>" "</s>" -n "{}"  "{}"  "{}" '.format(
            self.X['bigram_txt'], self.X['wdlist_txt'], self.X['biwnet_txt']
        ))

    def lm_build(self, order=2):
        HTK().readMLF(self.X['answer_mlf'], ['sil', 'sp']).writeMLF(self.X['inlang_mlf'], ['lm'])
        os.system('ngram-count -text "{}"  -write "{}"  -order "{}"  -vocab "{}" '.format(
            self.X['inlang_mlf'], self.X['wrdcnt_txt'], str(order), self.X['dictry_dct']
        ))
        os.system('ngram-count -read "{}"  -order "{}"  -vocab "{}"  -lm "{}" '.format(
            self.X['wrdcnt_txt'], str(order), self.X['dictry_dct'], self.X['outlan_txt']
        ))

    def lm_dict(self):
        HTK() \
            .readDCT(self.X['dictry_dct'], ['sil', 'sp']) \
            .writeDCT(self.X['landct_dct'])
        B = open(self.X['landct_dct'])
        text = B.readlines()
        A = open(self.X['landct_dct'], 'w')
        A.write('<s> sil\n</s> sil\n')
        for line in text:
            A.write(line)


    def bi_testing(self, opt_list=()):
        t0 = time.time()

        self.bigram2wordnet()

        os.system(
            'HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['wavlst_scp'], self.X['config_cfg'],
                self.X['biwnet_txt'], self.X['result_mlf'],
                self.X['landct_dct'], self.X['modelp_dct']
            ))

        os.system(
            'HResults -e "???" "<s>" -e "???" "</s>" -e "???" sil -e "???" sp -I "{}"  "{}"  "{}"  >> "{}" '.format(
                self.X['answer_mlf'], self.X['modelp_dct'],
                self.X['result_mlf'], self.X['accrcy_txt']
            ))
        self.report(t0)

    def exp_bi_testing(self, wav_list, result_mlf, nbest):
        self.feedback()
        self.bigram2wordnet()

        os.system(
            'HVite -D -n 4 "{}"  -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(
                nbest,
                self.X['macros_hmm'], self.X['models_hmm'],
                wav_list, self.X['config_cfg'],
                self.X['biwnet_txt'], result_mlf,
                self.X['landct_dct'], self.X['modelp_dct']
            ))

    def lat_testing(self, opt_list=()):
        t0 = time.time()
        self.bigram2wordnet()
        os.system(
            'HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l  "{}"  -i "{}"  -p 0.0 -s 0.0 -z txt -n 5 5 "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['wavlst_scp'], self.X['config_cfg'],
                self.X['biwnet_txt'], self.X['result_mlf'][:-11],
                self.X['result_mlf'],
                self.X['landct_dct'], self.X['modelp_dct']
            ))
        self.report(t0)

    def exp_testing(self, wav_list, result_mlf, nbest):
        os.system('HParse "{}"  "{}" '.format(self.X['grmmri_dct'], self.X['wdneti_dct']))
        os.system('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'], self.X['wdnetp_dct']))
        os.system(
            'HVite -D -n 4 "{}"  -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(
                nbest,
                self.X['macros_hmm'], self.X['models_hmm'],
                wav_list, self.X['config_cfg'],
                self.X['wdnetp_dct'], result_mlf,
                self.X['dictry_dct'], self.X['modelp_dct']
            ))

    def external_testing(self, wav_list, result_mlf):
        os.system('HParse "{}"  "{}" '.format(self.X['grmmri_dct'], self.X['wdneti_dct']))
        os.system('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'], self.X['wdnetp_dct']))
        os.system(
            'HVite -D  -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                wav_list, self.X['config_cfg'],
                self.X['wdnetp_dct'], result_mlf,
                self.X['dictry_dct'], self.X['modelp_dct']
            ))

    def testing(self, opt_list=()):
        t0 = time.time()

        os.system(
            'HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l \'*\' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" '.format(
                self.X['macros_hmm'], self.X['models_hmm'],
                self.X['wavlst_scp'], self.X['config_cfg'],
                self.X['wdnetp_dct'], self.X['result_mlf'],
                self.X['dictry_dct'], self.X['modelp_dct']
            ))
        """wdnet shortpause is too large to generate
        os.system("HVite -D -H "{}"  -H "{}"  -S "{}"  -C "{}"  -w "{}"  -l '*' -i "{}"  -p 0.0 -s 0.0 "{}"  "{}" ".format(
            self.X['macros_hmm'], self.X['models_hmm'], 
            self.X['wavlst_scp'], self.X['config_cfg'], 
            self.X['wdneti_dct'], self.X['result_mlf'], 
            self.X['dictry_dct'], self.X['modelp_dct']
        ))
        """
        os.system(
            'HResults -e "???" "<s>" -e "???" "</s>" -e "???" sil -e "???" sp -I "{}"  "{}"  "{}"  >> "{}" '.format(
                self.X['answer_mlf'], self.X['modelp_dct'],
                self.X['result_mlf'], self.X['accrcy_txt']
            ))
        self.report(t0)

    def am_setup(self):
        HTK() \
            .readMLF(self.dump) \
            .convertMLF2DCT() \
            .writeMLF(self.X['result_mlf'], ['sil', 'sp']) \
            .writeDCT(self.X['dictry_dct'], ['sil', 'sp'])

    def init_setup(self):
        HTK() \
            .readMLF(self.dump) \
            .writeMLF(self.X['result_mlf'], ['sil', 'sp']) \
            .writeMLF(self.X['answer_mlf']) \
            .writeMLF(self.X['phonei_mlf'], ['phone', 'sil']) \
            .writeMLF(self.X['phonep_mlf'], ['phone', 'sil', 'sp']) \
            .convertMLF2DCT() \
            .writeDCT(self.X['dictry_dct'], ['sil', 'sp']) \
            .writeDCT(self.X['modeli_dct'], ['model', 'sil']) \
            .writeDCT(self.X['modelp_dct'], ['model', 'sp', 'sil']) \
            .writeDCT(self.X['grmmri_dct'], [self.sil_tag, 'sil']) \
            .writeDCT(self.X['grmmrp_dct'], [self.sil_tag, 'sp', 'sil'])

        os.system('HParse "{}"  "{}" '.format(self.X['grmmri_dct'], self.X['wdneti_dct']))
        os.system('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'], self.X['wdnetp_dct']))

        self.training()
        self.testing()

    def feedback(self):
        HTK() \
            .readMLF(self.X['result_mlf'], ['sp', 'sil', '<s>', '</s>']) \
            .readDCT(self.X['dictry_dct']) \
            .writeMLF(self.X['answer_mlf']) \
            .writeMLF(self.X['phonei_mlf'], ['phone', 'sil']) \
            .writeMLF(self.X['phonep_mlf'], ['phone', 'sil', 'sp']) \
            .convertMLF2DCT() \
            .writeDCT(self.X['dictry_dct'], ['sil', 'sp']) \
            .writeDCT(self.X['modeli_dct'], ['model', 'sil']) \
            .writeDCT(self.X['modelp_dct'], ['model', 'sp', 'sil']) \
            .writeDCT(self.X['grmmri_dct'], [self.sil_tag, 'sil']) \
            .writeDCT(self.X['grmmrp_dct'], [self.sil_tag, 'sp', 'sil'])

        os.system('HParse "{}"  "{}" '.format(self.X['grmmri_dct'], self.X['wdneti_dct']))
        os.system('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'], self.X['wdnetp_dct']))

    def feedback_threshold0(self):
        HTK() \
            .readMLF(self.X['result_mlf'], ['sp', 'sil', '<s>', '</s>']) \
            .readDCT(self.X['dictry_dct']) \
            .writeMLF(self.X['answer_mlf']) \
            .writeMLF(self.X['phonei_mlf'], ['phone', 'sil']) \
            .writeMLF(self.X['phonep_mlf'], ['phone', 'sil', 'sp']) \
            .convertMLF2DCT(0) \
            .writeDCT(self.X['dictry_dct'], ['sil', 'sp']) \
            .writeDCT(self.X['modeli_dct'], ['model', 'sil']) \
            .writeDCT(self.X['modelp_dct'], ['model', 'sp', 'sil']) \
            .writeDCT(self.X['grmmri_dct'], [self.sil_tag, 'sil']) \
            .writeDCT(self.X['grmmrp_dct'], [self.sil_tag, 'sp', 'sil'])

        os.system('HParse "{}"  "{}" '.format(self.X['grmmri_dct'], self.X['wdneti_dct']))
        os.system('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'], self.X['wdnetp_dct']))


    def lexicon_setup(self):
        # self.recognize_patterns()
        import suffix
#        suffix.flatten_pattern(self.X['dictry_dct'],self.X['result_mlf'],self.X['flattn_dct'],self.X['flattn_mlf'])
#        suffix.parse_pattern(self.X['flattn_dct'],self.X['flattn_mlf'],self.X['dictry_dct'])
        suffix.flatten_pattern(self.X['dictry_dct'], self.X['result_mlf'], self.X['dictry_dct'], self.X['result_mlf'])
        suffix.parse_pattern(self.X['dictry_dct'], self.X['result_mlf'], self.X['dictry_dct'])

        HTK() \
            .readDCT(self.X['dictry_dct'], ['sil', 'sp']) \
            .writeDCT(self.X['modeli_dct'], ['model', 'sil']) \
            .writeDCT(self.X['modelp_dct'], ['model', 'sil', 'sp']) \
            .writeDCT(self.X['grmmri_dct'], [self.sil_tag, 'sil']) \
            .writeDCT(self.X['grmmrp_dct'], [self.sil_tag, 'sp', 'sil'])

        os.system('HParse "{}"  "{}" '.format(self.X['grmmri_dct'], self.X['wdneti_dct']))
        os.system('HParse "{}"  "{}" '.format(self.X['grmmrp_dct'], self.X['wdnetp_dct']))
        print "finish lexicon_setup"

    def iteration(self, command, comment='', do_slice=False):
        self.time = time.time()
        self.record_comment('{}_{}'.format(str(self.offset + 1), comment))
        if command == 'a': self.a()
        if command == 'a_mix': self.a_mix()
        if command == 'a_keep': self.a_keep()
        if command == 'al': self.al()
        if command == 'al_lat': self.al_lat()
        if command == 'x': self.x()
        if command == 'ax': self.ax()
        if do_slice:  self.slice()
        self.record_time()
        self.save_snapshot(comment)
    def initialization(self, comment=''):
        self.clean()
        # self.matlab()
        self.feature()

        # self.am_setup() #obsolete, bad for large corpus
        self.init_setup()
        self.save_snapshot(comment)
    def save_snapshot(self, comment):
        if self.do_copy:
            os.chdir(self.target)
            os.chdir('..')
            if self.target[-1]=='/':
                name = self.target.split('/')[-2]
            else:
                name = os.path.basename(self.target)
            if comment:
                self.writeASR('{}_snapshot/{}_{}_{}/'.format(name, str(self.offset + 1),name, comment))
            else:
                self.writeASR('{}_snapshot/{}_{}/'.format(name, str(self.offset + 1),name))
        self.offset += 1

        #os.chdir(self.target)
        #os.chdir('..')
        #self.writeASR('0_{}/'.format(comment))

    def a(self):
        print "a feedback"
        self.feedback()
        print "a training"
        self.training()
        print "a testing"
        self.testing()

    def a_mix(self):
        print "a_mix feedback"
        self.feedback()
        print "a_mix training(addmix)"
        self.training('addmix')
        print "a testing"
        self.testing()

    def a_keep(self):
        print "a_keep feedback"
        self.feedback()
        print "a_keep training(keepmix)"
        self.training('keepmix')
        print "a_keep testing"
        self.testing()

    def al(self):
        print "al feedback_threshold0"
        self.feedback_threshold0()
        print "al training"
        self.training()
        print "al lm_build"
        self.lm_build()
        print "al bi_testing"
        self.bi_testing()

    def al_lat(self):
        self.feedback()
        self.training('keepmix')
        self.lat_testing()

    def x(self):
        self.feedback()
        self.lexicon_setup()
        self.testing()

    def ax(self):
        print "ax feedback"
        self.feedback()
        print "ax training"
        self.training()
        print "ax lexicon setup"
        self.lexicon_setup()
        print "ax testing" 
        self.testing()

    '''
    def Four_Phase(self):
        self.initialization()
        while True:
            self.iteration('a','acoustic_iteration') 
            if evaluation.pass_consistency(self): break
        while True:
            self.iteration('al','linguistic_iteration') 
            if evaluation.pass_consistency(self): break
        while True:
            self.iteration('ax','lexical_iteration') 
            if evaluation.pass_consistency(self): break
    '''
