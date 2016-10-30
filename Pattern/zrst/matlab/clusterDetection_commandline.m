%function clusterDetection(path)


%path = 'C:/Users/Tao/Dropbox/Semester 9.5/Corpus_TIMIT_train/'
%path = 'C:/Users/Tao/Dropbox/Semester 9.5/Corpus_TIMIT_dev/'
%path = 'C:/Users/Tao/Dropbox/Semester 8.5/Corpus_5034wav/'
%dumpfile = 'IDump_timit_train_100.txt'
%dumpfile = 'IDump_5034wav_100.txt'
%dumpfile = 'IDump_timit_dev_50.txt'
%wavPath = 'wav_165/';
%wavPath = 'news/';    %true, 100,5,0,
%wavPath = 'wav_13/';%false,30,5,0.2
%wavPath = 'twilight/';
%wavPath = '../../Corpus_digit333/'
wavPath = path

iter=4;%4
powerorder=4;%4
peakdelta=0.2;%0.2

normalize = false;%false
%clusterNumber = 50;%30
frameThresh = 5%10,5
energyThresh = 0.2%0.3,0.2

wavList = dir(wavPath);
wavList = wavList(3:length(wavList));
%'../../Corpus_pcm/N200108011200-30-04.pcm'

% temp=1;
% while 1
%     wavName = wavList(temp).name;
%     if ~strcmp(wavName(length(wavName)-3:length(wavName)),'.wav')
%         wavList(temp)=[];
%     else
%         temp=temp+1;
%     end
%     if temp==length(wavList)+1
%         break;
%     end
% end

w = length(wavList);
featureId = 1;
F =[];
for y = 1:w
    clear M Audio Memph cut I Idist Iwater N Ncut 
    wavName = wavList(y).name
    wav = wavName(1:length(wavName)-4);
    M=getfeature([wavPath wavName]);
    %[Audio fs] = wavread([wavPath wavName]);
    
    %[a temp ] = size(Audio);
    [m temp ] = size(M);
    
    

    Memph =M(:,13).^powerorder;

    [MAXTAB, MINTAB] = peakdet(Memph,peakdelta);
    if ~isempty(MINTAB)
        cut = [1; MINTAB(:,1); m];
    else
        cut = [1;  m];
    end
    c = length(cut);

    for i =1:c-1
        N{i} = M(cut(i):cut(i+1),:);
    end


    for x=1:c-1
        I = dotplot(N{x},N{x});
        Idist{1}=I;

        for i =1:iter-1
            Idist{i+1} = im2bw(Idist{i},graythresh(Idist{i}));
            Idist{i+1} = bwdist(~Idist{i+1});
            Idist{i+1} = Idist{i+1}./max(max(Idist{i+1}));

        end
		%#imshow(Idist{1})
        Iwater = watershed(1-Idist{iter});
        Isub = find(diag(Iwater)==0);

        Ncut = [1;Isub;length(Iwater)];
        Nc = length(Ncut);
        
        ireal=1;
        for i =1:Nc-1
            fbeg = Ncut(i)   + cut(x)-1;
            fend = Ncut(i+1) + cut(x)-1;
            feature = mean( M(fbeg:fend,:) );
            if normalize
                feature = feature./sqrt(sum(feature.^2));
            end
            if fend-fbeg > frameThresh && feature(13) > energyThresh
                %FId{y}{x}{ireal}.abeg = floor(fbeg*a/m);
                %FId{y}{x}{ireal}.aend = floor(fend*a/m);
                %mkdir(['x'])
                %wavwrite(Audio(floor(fbeg*a/m):floor(fend*a/m)),fs,['x\' wav
                %'_' num2str(x) '_' num2str(i) '.wav'])
                F = [F;  feature];
                FId{y}{x}{ireal}.featureId = featureId;
                featureId = featureId+1;
                ireal=ireal+1;
            end
        end
    end
    fclose('all');
end

%imshow(dotplot(F,F))
%clusterTable = clusterdata(F,100);

%Taipei
clusterTable = kmeans(F,clusterNumber);
%Taichung
%[temp1 temp2 clusterTable] = kmeans(F,clusterNumber);

[f temp]=size(F);
% clusterNum = 0;
% 
% 
% for i =1:f 
%     feature = F(i,:);
%     
%     feature = feature./sqrt(sum(feature.^2));
%     maxValue = 0;    
%     maxId = j;
%     for j =1:clusterNum
%         %cosValue = abs(clusterList{j}.avg.*feature)
%         cosValue = sum(clusterList{j}.avg.*feature);
%         if cosValue > maxValue
%             maxValue = cosValue;
%             maxId = j;
%         end
%     end
%     if maxValue > 0.7
%         s = clusterList{maxId}.size;
%         clusterList{maxId}.avg = (clusterList{maxId}.avg * s + feature) / (s+1);
%         %clusterList{j}.list = [clusterList{j}.list; feature]
%         clusterList{maxId}.size = clusterList{maxId}.size + 1 ;
%         clusterTable(i) = maxId;
%     else
%         clusterNum = clusterNum + 1
%         clusterList{clusterNum}.avg = feature;
%         %clusterList{j}.list = [feature]
%         clusterList{clusterNum}.size = 1;
%         clusterTable(i) = clusterNum;
%     end
% end
% 

clusterDir = [wavPath(1:length(wavPath)-1) '_cluster/'];
%rmdir(clusterDir,'s');
%mkdir(clusterDir);

for y = 1:w
    clear M Audio Memph cut I Idist Iwater N Ncut 
    wavName = wavList(y).name
    wav = wavName(1:length(wavName)-4);
    %[Audio fs] = wavread([wavPath wavName]);
    for x =1:length(FId{y})
        for i =1:length(FId{y}{x})
            clusterId = clusterTable(FId{y}{x}{i}.featureId);
            FId{y}{x}{i}.clusterId = clusterId;
            %abeg = FId{y}{x}{i}.abeg;
            %aend = FId{y}{x}{i}.aend;
            %wavwrite(Audio(abeg:aend),fs,[clusterDir num2str(clusterId) '_' wav '_' num2str(x) '_' num2str(i) '.wav']); 
        end
    end

    fclose('all');
end

%textfile = fopen([wavPath(1:length(wavPath)-1) '.txt'],'w')
textfile = fopen([ dumpfile],'w');
for y = 1:w
    fprintf(textfile,'%s\n', wavList(y).name);
    for x =1:length(FId{y})
        for i =1:length(FId{y}{x})
            fprintf(textfile,'%d ', FId{y}{x}{i}.clusterId);
            %fprintf(textfile,'%d\n', FId{y}{x}{i}.clusterId);
        end
        fprintf(textfile,'\n');
    end
end

fclose('all');
exit;
%imshow(dotplot(F,F));
