import os
import shutil

df_model='test100_obama'

def handle_filename(name):
    name = name.replace("\"","").replace("*","").replace("/","")
    #print name
    name = name.split('-')
    filename = name[0][5:]
    sent_n = name[1][:3]
    #print '{} {}'.format(filename,sent_n)
    return filename,sent_n


def main(model=df_model):

    mlf_path='Results/' + model + '/result/result.mlf'
    out_path='Results/' + model + '/result_sentence/'

    sent = ''
    flag = False
    out = {}
    filename = ''
    sent_n = ''

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    #open file
    with open(mlf_path,'r') as f:
        next(f)#skip first line which is "#!MLF!#"
        for line in f:
            line = line.split()#400000 2200000 p29p1 -1081.670532

            #check if this line is filename
            if len(line) == 1 and not flag:
                filename , sent_n = handle_filename(line[0])
                flag = True
                continue

            #check if this line is end of the sentence
            elif len(line)==1 and flag:
                if filename not in out:
                    out[filename] = {}
                out[filename][sent_n]=sent
                sent = ''
                flag = False
                continue

            #add word to sentence
            if line[2] != 'sp' and line[2] !='sil':
                sent += line[2]+' '

    for doc in out:
        f = open(out_path+doc+'.txt','w')
        print doc
        for s in sorted(out[doc]):
            print s
            print out[doc][s]
            f.write(s+' '+out[doc][s]+'\n')

if __name__ == '__main__':
    main()
