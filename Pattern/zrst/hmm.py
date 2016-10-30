import math

import numpy as np


def not_zero(x):
    eps = 1e-99
    x = np.array(x)
    x[x < eps] = eps
    return x


def np_vector(x):
    return not_zero(map(float, x.split()))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def mylog(x):
    try:
        return math.log(x)
    except:
        return -999.9


def parseBI(bi):
    lines = open(bi).readlines()
    id2word = dict({})
    bigram = dict({})
    null_list_end = dict({})
    null_list_beg = dict({})
    null_id = 0
    for line in lines:
        line = line.strip('\n')
        if line[0] == 'I':
            id = int(line.split()[0].split('=')[1])
            word = line.split()[1].split('=')[1]
            id2word[id] = word

        if line[0] == 'J':
            begi, endi = int(line.split()[1].split('=')[1]), int(line.split()[2].split('=')[1])
            beg, end = id2word[begi], id2word[endi]
            log10likelihood = float(line.split()[3].split('=')[1])
            if begi == null_id:
                null_list_end[end] = log10likelihood
            if endi == null_id:
                null_list_beg[beg] = log10likelihood
                continue
            bigram[beg, end] = log10likelihood
    for beg in null_list_beg:
        for end in null_list_end:
            if (beg, end) in bigram:
                bigram[beg, end] += null_list_beg[beg] + null_list_end[end]
            else:
                bigram[beg, end] = null_list_beg[beg] + null_list_end[end]
    return bigram


# B = parseBI('64_l5034_lab/lm/biwnet.txt')
# print B

def parseLEX(lex):
    lines = open(lex).readlines()
    lex = dict({})
    for line in lines:
        line = line.strip('\n')
        lex[line.split()[0]] = line.split()[1:]
    return lex


# L = parseLEX('64_l5034_lab/library/dictionary.txt')
# print L


def parse_hmm(model):
    lines = open(model).readlines()
    M = open(model)

    def getint(string):
        return int(string.split()[1])

    hmm = dict({})
    model_name = ''
    mode = ''
    gauss = ''
    for ___ in lines:
        line = M.readline()
        line = line.strip('\n')
        if '~h' in line:
            model_name = line.split()[1][1:-1]
            hmm[model_name] = dict({})
            continue
        if not model_name: continue
        if '<NUMSTATES>' in line:
            num_state = getint(line)
            hmm[model_name]['num_state'] = num_state
        elif '<STATE>' in line:
            state = getint(line)
            hmm[model_name][state] = dict({})
        elif '<NUMMIXES>' in line:
            num_gauss = getint(line)
            hmm[model_name][state]['num_gauss'] = num_gauss
        elif '<MIXTURE>' in line:
            gauss = getint(line)
            weight = float(line.split()[2])
            hmm[model_name][state][gauss] = dict({})
            hmm[model_name][state][gauss]['weight'] = weight
        elif '<MEAN>' in line:
            mode = 'mean'
        elif '<VARIANCE>' in line:
            mode = 'variance'
        elif '<TRANSP>' in line:
            assert int(line.split()[1]) == hmm[model_name]['num_state']
            hmm[model_name]['trans'] = dict({})
            mode = 'trans'
            trans_row = 0
        else:
            if mode == 'mean':
                try:
                    hmm[model_name][state][gauss]['mean'] = np_vector(line)
                    #hmm[model_name][state][gauss]['mean'] = line.split()
                except:
                    hmm[model_name][state]['num_gauss'] = 1
                    gauss, weight = 1, 1
                    hmm[model_name][state][gauss] = dict({})
                    hmm[model_name][state][gauss]['weight'] = weight
                    hmm[model_name][state][gauss]['mean'] = np_vector(line)
                mode = ''
            elif mode == 'variance':
                hmm[model_name][state][gauss]['variance'] = np_vector(line)
                mode = ''
            elif mode == 'trans':
                hmm[model_name]['trans'][trans_row] = np_vector(line)
                trans_row += 1
                if trans_row == hmm[model_name]['num_state'] - 1:   mode = ''
    return hmm


#H = parseHMM('64_l5034_lab/hmm/models')
#pprint(H['p1'])

def kld_gss(A, B):
    varA = A['variance']
    varB = B['variance']
    invB = 1.0 / varB
    vAiB = varA * invB
    term1 = np.sum(vAiB)

    uA = A['mean']
    uB = B['mean']
    ud = uB - uA
    term2 = np.sum(ud * invB * ud)
    term3 = np.sum(np.log(varB) - np.log(varA))
    term4 = -float(len(uA))
    return 0.5 * (term1 + term2 + term3 + term4)


'''
H = parse_hmm(/home/c2tao/Dropbox/Semester 12.5/ICASSP 2015 Data/5034_relabeled_active/hmm/models')
g1 = H['p1'][2][1]
g2 = H['p1'][2][2]
g3 = H['p2'][2][1]
print kld_gss(g1,g1)
print kld_gss(g1,g2)
print kld_gss(g1,g3)
'''


def kld_gmm(A, B):
    #nA = A['num_gauss']
    #nB = B['num_gauss']
    nA, nB = len(A) - 1, len(B) - 1

    def skip_index(A):
        vA = sorted(A.keys())
        vA.remove('num_gauss')
        vA.insert(0, 0)  #insert dummy index
        return vA

    vA, vB = skip_index(A), skip_index(B)
    #print vA,vB

    term = []
    if nA == 1 and nB == 1:
        return kld_gss(A[vA[1]], B[vB[1]])
    for j in range(1, nA + 1):
        a = A[vA[j]]

        top = not_zero([kld_gss(a, A[vA[i]]) for i in range(1, nA + 1)])
        w_top = not_zero([A[vA[i]]['weight'] for i in range(1, nA + 1)])
        bot = not_zero([kld_gss(a, B[vB[i]]) for i in range(1, nB + 1)])
        w_bot = not_zero([B[vB[i]]['weight'] for i in range(1, nB + 1)])

        term1 = np.log(np.sum(w_top * np.exp(-top)))
        term2 = np.log(np.sum(w_bot * np.exp(-bot)))
        #print term1,term2
        term.append(a['weight'] * (term1 - term2))
    return 1.0 * sum(term)


'''
H = parse_hmm(/home/c2tao/Dropbox/Semester 12.5/ICASSP 2015 Data/5034_relabeled_active/hmm/models')
m1 = H['p1'][2]
m2 = H['p1'][3]
m3 = H['p2'][2]
print kld_gmm(m1,m1)
print kld_gmm(m1,m2)
print kld_gmm(m1,m3)
'''


def kld_bid(A, B):
    return (kld_gmm(A, B) + kld_gmm(B, A)) / 2.0


def kld_hmm(A, B):
    assert (A.keys() == B.keys())
    l = 0.0
    distance = 0.0
    for k in A.keys():

        if is_number(k):
            l += 1.0
            distance += kld_bid(A[k], B[k])
            #print k, kld_bid(A[k],B[k])
    return distance / l


'''
H = parse_hmm(/home/c2tao/Dropbox/Semester 12.5/ICASSP 2015 Data/5034_relabeled_active/hmm/models')
h1 = H['p1']
h2 = H['p2']
h3 = H['p3']
print kld_hmm(h1,h1)
print kld_hmm(h1,h2)
print kld_hmm(h1,h3)
'''

