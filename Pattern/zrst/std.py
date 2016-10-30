import os
import shutil
import cPickle as pickle

import numpy as np

import util
import hmm


class STD(object):
    def __init__(self, root, label, qlist='', corpus=''):
        self.root = root
        self.label = label
        self.qlist = qlist
        self.corpus = corpus

        self.qlist_file = root + r'/qlist.txt'
        self.label_file = root + r'/label.mlf'

        self.pattern_list = []
        self.pattern_dict = {}

        self.feature_fold = root + r'/feature_file/'
        self.feature_file = root + r'/feature_list.scp'

        self.distanc_fold = root + r'/pattern_distance/'
        self.distanc_file = {}
        self.kl = []

        self.decoded_fold = root + r'/pattern_decoded/'
        self.decoded_file = {}

        self.similar_fold = root + r'/similarity/'
        self.similar_file = {'feature': self.similar_fold + 'feature.score'}

        self.query = []
        self.query_mlf = None
        self.query_answer = {}
        self.query_answer_file = root + r'/query_answer'

        self.similarity = {}

    def query_init(self):
        self.query_mlf = util.MLF(self.label)

        try:
            os.mkdir(self.feature_fold)
        except:
            True
        try:
            os.mkdir(self.distanc_fold)
        except:
            True
        try:
            os.mkdir(self.decoded_fold)
        except:
            True
        try:
            os.mkdir(self.similar_fold)
        except:
            True

        if not os.path.exists(self.label):
            shutil.copyfile(self.label, self.label_file)
        if not os.path.exists(self.qlist):
            shutil.copyfile(self.qlist, self.qlist_file)

        if not os.path.exists(self.feature_file):
            files = util.make_feature(self.corpus, self.feature_fold)
            with open(self.feature_file, 'w') as myfile:
                for f in files:
                    myfile.write(f + '\n')
            myfile.close()


    def parse_query(self):
        self.query = map(lambda x: tuple(x.strip().split()), open(self.qlist, 'r').readlines())
        if os.path.exists(self.query_answer_file):
            self.query_answer = pickle.load(open(self.query_answer_file, 'rb'))
            return self.query_answer


        # self.query_answer = np.zeros((len(self.query), len(self.query_mlf.wav_list)))

        for j in range(len(self.query)):
            self.query_answer[j] = []
            for i in range(len(self.query_mlf.wav_list)):
                S = util.SubDTW(self.query_mlf.tag_list[i], self.query[j], lambda x, y: 0 if x == y else 1)
                # print S.calculate()
                # print self.query[j]
                # print self.query_mlf.tag_list[i]
                if S.calculate() == 0:
                    p = S.get_path()
                    #print p 
                    if p[-1][0] == 0:
                        tbeg = 0
                    else:
                        tbeg = self.query_mlf.int_list[i][p[-1][0] - 1]
                    tend = self.query_mlf.int_list[i][p[0][0]]
                    #self.query_answer[j] += (self.query_mlf.wav_list[i],tbeg,tend),
                    self.query_answer[j] += (i, tbeg, tend),
                    #print (self.query_mlf.wav_list[i],tbeg,tend)
        pickle.dump(self.query_answer, open(self.query_answer_file, 'wb'))
        return self.query_answer

    def feature_query(self, query_index, inst_index):
        w_index, t_beg, t_end = self.query_answer[query_index][inst_index]
        # print self.feature_fold + self.query_mlf.wav_list[w_index]
        f_file = self.feature_fold + self.query_mlf.wav_list[w_index] + '.mfc'
        return util.read_feature(f_file)[t_beg:t_end]

    def feature_dtw(self, query_index, inst_index):
        Q = self.feature_query(query_index, inst_index)
        similarity = []
        for ___ in range(len(self.query_mlf.wav_list)):
            similarity.append(float('Inf'))
        for w in range(len(self.query_mlf.wav_list)):
            f_file = self.feature_fold + self.query_mlf.wav_list[w] + '.mfc'
            D = util.read_feature(f_file)
            similarity[w] = util.warp(util.cos_dist(D, Q))
            # similarity[w] = util.warp(- np.dot(D,Q.T))
            print query_index, w, similarity[w]
        return similarity

    def feature_similarity(self):
        if os.path.exists(self.similar_file['feature']):
            return pickle.load(open(self.similar_file['feature'], 'rb'))
        similarity = {}
        for i in range(len(self.query)):
            similarity[i] = self.feature_dtw(i, 0)
        pickle.dump(similarity, open(self.similar_file['feature'], 'wb'))
        return similarity

    def pattern_query(self, query_index, inst_index, pattern_name):
        # print self.decoded_file
        mlf = util.MLF(self.decoded_file[pattern_name])
        w_index, t_beg, t_end = self.query_answer[query_index][inst_index]
        return mlf.wav_dur(w_index, t_beg, t_end)

    def pattern_dtw(self, query_index, inst_index, pattern_name, distance_metric):
        query_seq = self.pattern_query(query_index, inst_index, pattern_name)

        D = util.MLF(self.pattern_dict[pattern_name].X['result_mlf'])
        similarity = []
        for ___ in range(len(self.query_mlf.wav_list)):
            similarity.append(float('Inf'))
        for w in range(len(D.wav_list)):
            similarity[w] = util.SubDTW(D.tag_list[w], query_seq, lambda a, b: distance_metric[(a, b)]).calculate()
            print pattern_name, query_index, w, similarity[w]
        return similarity

    def pattern_similarity(self, pattern_name):
        if os.path.exists(self.similar_file[pattern_name]):
            return pickle.load(open(self.similar_file[pattern_name], 'rb'))
        similarity = {}
        dm = self.get_dm(pattern_name)
        for i in range(len(self.query)):
            similarity[i] = self.pattern_dtw(i, 0, pattern_name, dm)
        pickle.dump(similarity, open(self.similar_file[pattern_name], 'wb'))
        return similarity

    def get_dm(self, pattern_name):
        dm = self.pattern_distanc(pattern_name)

        ok = set([])
        for k in dm.keys():
            ok.add(k[0])
            ok.add(k[1])
        ok = list(ok)

        for k in ok:
            ambig = np.mean([dm[x] for x in dm.keys() \
                             if (x[0] == k or x[1] == k) and dm[x] != float('Inf')])
            dm[('sil', k)] = ambig
            dm[('sp', k)] = ambig
        ambig = np.mean([dm[x] for x in dm.keys() \
                         if dm[x] != float('Inf')])
        dm[('sil', 'sp')] = ambig
        dm[('sil', 'sil')] = ambig
        dm[('sp', 'sp')] = ambig

        for k in dm.keys():
            dm[(k[1], k[0])] = dm[k]
        return dm

    def mean_average_precision(self, similarity):
        mean_ap = []
        q_answer = [0.0 for i in range(len(self.query_mlf.wav_list))]
        for i in self.query_answer:
            for j in range(len(self.query_answer[i])):
                q_answer[self.query_answer[i][j][0]] = 1.0

            # mean_ap += util.average_precision(q_answer,np.array(similarity[i])),
            mean_ap += util.average_precision_minus1(q_answer, np.array(similarity[i])),
        return mean_ap

    def query_build(self):
        for p in self.pattern_list:
            self.pattern_distanc(p)
            self.pattern_decoded(p)

    def add_pattern(self, pattern, pattern_name):
        self.pattern_list += pattern_name,
        self.pattern_dict[pattern_name] = pattern
        self.decoded_file[pattern_name] = self.decoded_fold + '/' + pattern_name + '.mlf'
        self.distanc_file[pattern_name] = self.distanc_fold + '/' + pattern_name + '.kl'
        self.similar_file[pattern_name] = self.similar_fold + '/' + pattern_name + '.score'

    def pattern_decoded(self, pattern_name):
        if os.path.exists(self.decoded_file[pattern_name]): return
        self.pattern_dict[pattern_name].external_testing(self.feature_file, self.decoded_file[pattern_name])

    def pattern_distanc(self, pattern_name):
        if os.path.exists(self.distanc_file[pattern_name]):
            return pickle.load(open(self.distanc_file[pattern_name], 'rb'))
        dm = {}
        H = hmm.parse_hmm(self.pattern_dict[pattern_name].X['models_hmm'])
        p_list = []
        for p in H.keys():
            if 's' not in p:
                p_list.append(p)

        for i in p_list:
            for j in p_list:
                if (j, i) not in dm:
                    dm[(i, j)] = hmm.kld_hmm(H[i], H[j])
        pickle.dump(dm, open(self.distanc_file[pattern_name], 'wb'))
        return pickle.load(open(self.distanc_file[pattern_name], 'rb'))

    def query_copy(self):
        '''
        only use this function if the query corpus is exactly the same as the pattern corpus
        this copies the result.mlf from the ASR objects
        '''
        for p in self.pattern_list:
            shutil.copyfile(self.pattern_dict[p].X['result_mlf'], self.decoded_file[p])


'''
dropbox_path = r'/home/c2tao/Dropbox/'
query_path = dropbox_path + r'Semester 12.5/ICASSP 2015 Data/5034_query_active/'
labels_path = dropbox_path + r'Semester 8.5/_readonly/homophone_time_missing.mlf'
corpus_path = dropbox_path + r'Semester 8.5/Corpus_5034wav/'
target_path1 = dropbox_path + r'Semester 9.5/40_5034_50_3/'
target_path2 = dropbox_path + r'Semester 9.5/36_5034_50_3/'

A = ASR(target = target_path1)
B = ASR(target = target_path2)
Q = STD(root = query_path, label = labels_path, corpus = corpus_path)
Q.add_pattern(A,'40_5034_50_3')
Q.add_pattern(B,'36_5034_50_3')
Q.query_init()
Q.query_build()
'''
