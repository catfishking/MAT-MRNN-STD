%input_path = 'C:/Users/Tao/Dropbox/Semester 9.5/Corpus_TIMIT_test/'
input_path = '../../Semester 10/TCCGMM/Corpus_TIMIT_train/'
output_path = '../../Semester 9.5/Corpus_TIMIT_train_MFCC/'
wavList = dir(input_path);
wavList = wavList(3:length(wavList));
w = length(wavList);


for y = 1:w
    wavName = wavList(y).name
    load([input_path wavName])
    
	fid = fopen([output_path wavName(1:end-8) '.mfc'], 'w');
    fwrite(fid, size(feature_dbn,1), 'int32');
    fwrite(fid, 100000, 'int32');
    fwrite(fid, size(feature_dbn,2)*4, 'int16');
	fwrite(fid, 9, 'int16');
    fwrite(fid, feature_dbn', 'single');
    fclose(fid);
end

%seed = feature('getpid')
%cmd=sprintf('HList.exe -C hcopy_dbn.cfg -i 39 "%s" > "%s"','test.mfc',['temp' int2str(seed) '.txt']);
%cmd=sprintf('HList.exe -C hcopy_scale.cfg -i 39 "%s" > "%s"',wave,'temp.txt');
%system(cmd);
