function [LT,LF,dT,dF,feat,win,parmKind] = read_feature(filename)
% Read feature from file
% [LT,LF,dT,dF,feat] = read_feature(filename)
% [LT,LF,dT,dF,feat] = read_feature(filename,frameshift)
% frameshift is default set to 4ms for htk-related files.

mfc_config_file = 'wav2mfc_0DAZ_4ms.cfg';
wav_file = regexprep(filename,'\.\w+','.wav');

ext = regexprep(filename,'.*\.','');
switch ext
case {'spec','feat'}
	filetype = 'spec';
	parmKind = 0;
	%if(~exist(filename,'file'))
	%	wav2spec(wav_file);
	%end
case {'mfc','fbank','plp'}
	filetype = 'htk';
	dF = 1;
	%if(~exist(filename,'file'))
	%	!HCopy -C mfc_config_file wav_file filename
	%end
	win = 8e-3;
case {'mat'}
	filetype = 'mat';
	dF = 1;
    dT = 1;
    parmKind = 0;
	%if(~exist(filename,'file'))
	%	!HCopy -C mfc_config_file wav_file filename
	%end
	win = 8e-3;
otherwise
	error('unknown extension: %s',ext);
end

switch filetype
	case 'spec'
		[LT,LF,dT,dF,feat,win] = read_spec(filename);
	case 'htk'
		[LT,LF,feat,dT,parmKind] = read_htk(filename);
	case 'mfc'
		[LT,LF,feat,dT,parmKind] = read_htk(filename);
    case 'mat'
        [LT,LF,feat] = read_mat(filename);
end
