function M=getfeature(wave)
seed = feature('getpid')
%cmd=sprintf('c:\\cygwin\\bin\\HList.exe -C hcopy.cfg -i 39 "%s" > "%s"',wave,'temp.txt');
cmd=sprintf('HList -C hcopy.cfg -i 39 "%s" > "%s"',wave,['temp' int2str(seed) '.txt']);
%cmd=sprintf('HList.exe -C hcopy_scale.cfg -i 39 "%s" > "%s"',wave,'temp.txt');
system(cmd);

fid=fopen(['temp' int2str(seed) '.txt']);
C = textscan(fid, ['%s'...      
    '%f %f %f %f %f    %f %f %f %f %f    %f %f %f' ...
    '%f %f %f %f %f    %f %f %f %f %f    %f %f %f' ...
    '%f %f %f %f %f    %f %f %f %f %f    %f %f %f' ...
],'HeaderLines',1);%skip firstline
n = length(C{1})-1;%skip last line
M = zeros(n,39);
for i=1:n
    for j=1:39
        M(i,j)=C{j+1}(i);
    end
end
