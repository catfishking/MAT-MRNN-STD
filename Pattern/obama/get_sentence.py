
from os import listdir
from os.path import isfile, join
import subprocess


def main(argv):
    
    #set path
    file_path = '/home/joe32140/obama_week_address/subtitle/'
    wav_path = '/home/joe32140/obama_week_address/wav/'
    out_path = '/tmp2/troutman/obama/sentence_wav/'
    filename_map = '/tmp2/troutman/obama/filename.map'
    transcripts_path='/tmp2/troutman/obama/sentence_transcripts/'

    myfilename_map = open(filename_map,'w')#filename map id file

    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]#for each files in wav_path directory

    file_count = 0
    total_length = 0
    for myfile in files:
        f = open(file_path + myfile,'r')#open wav file
        f_t = open(transcripts_path+str(file_count).zfill(3)+'.txt','w')#open sentence transcript text file
        begin = 0.0
        end = 0.0
        duration = 0.0
        flag = False
        sentence = ''
        wav_name = f.name.split('/')[-1]#get wav file name
        myfilename_map.write(str(file_count).zfill(3)+' '+wav_name+'\n')
        
        f_t.write(wav_name+'\n')

        wav_name = wav_path + wav_name.replace(' ','\ ').replace('\'','\\\'')[:-6]+'wav'
        count = 0#count sentence number
        for line in f:
            line =  line.replace('\n','')
            if(len(line) > 0):
                if( line.find('-->') >= 0):
                    time = line.replace('-->',':')
                    
                    time = time.split(':')
                    if( not flag):
                        begin = 3600*float(time[0]) + 60 * float(time[1]) + float(time[2])
                        flag = True
     #               print begin
                    end = 3600*float(time[3]) + 60*float(time[4]) + float(time[5])

                elif(begin != 0):
                    sentence += line + " "

                if( line[-1] == '.'):
                    duration = end - begin
                    flag = False
                    bash_command =  "sox " + wav_name + ' '+\
                                    out_path+'obama'+str(file_count).zfill(3)+'-'+str(count).zfill(3)+'.wav'\
                                    ' trim ' + str(begin) + ' ' + str(duration)
                    print bash_command
                    subprocess.Popen(bash_command,shell = True)
                    f_t.write(str(count).zfill(3) + " " + sentence + '\n')
                    total_length += duration
                    count += 1
                    sentence = ""
        file_count += 1
        f_t.close()

    print "total minutes:{}".format(total_length)
if __name__ == "__main__":
    import sys
    main(sys.argv)
