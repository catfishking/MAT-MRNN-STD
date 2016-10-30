function [boundary,Segments,basic_seg_idx,seg_thresh,Tree] = seg_by_hac(filename, varargin)
% boundary = seg_by_hac(filename, options)
% seg_by_hac segment the file by HAC with one of the following criterions:
% 1) minimum total variance (mintotvar) or
% 2) minimum linear regression error (minlinerr)
% example: seg_by_hac(fbank_file, 'isdemo', 'mintotvar');
%          seg_by_hac(fbank_file, 'isdemo', 'minlinerr');


plot_width = 10;
threshold_parm1 = 2;
LRwin = 5;
nband = 5;

okargs = lower({'isdemo', 'openwave', 'mintotvar','minlinerr'});

isdemo = false;
openwave = false;
type = 0;

% Parse arguments%{{{
if nargin > 1
  j = 1;
  while j <= length(varargin)
    pname = varargin{j};
    k = strmatch(lower(pname), okargs, 'exact');
    if isempty(k)
      error('stats:dtw_seg_tree:BadParameter', ...
      'Unknown parameter name: %s.',pname);
    elseif length(k) > 1
      error('stats:dtw_seg_tree:BadParameter', ...
      'Ambiguous parameter name: %s.',pname);
    else
      switch k
        case 1  % isdemo
          isdemo = true;
        case 2  % openwave
          openwave = true;
        case 3  % mintotvar
          type = 1;
        case 4  % minlinerr
          type = 2;
      end
    end
    j = j + 1;
  end
end
%}}}

if isdemo
  disp(filename);
end

if type == 0
  error('boundary = seg_by_hac(filename, options) must contain ''mintotvar'' or ''minlinerr'' in options');
end


[LT,LF,dT,dF,feat,winsize] = read_feature(filename);
F = (0:(LF-1)) * dF;
T = winsize/2 + (0:LT-1) * dT;
plot_width = min(plot_width,T(end));

original_feat = feat;
%feat = down_sample(feat(1:end-1,:), nband);
LF = size(feat, 1);

% Load wavefile%{{{
if (isdemo && openwave)
  wav_file = regexprep(filename,'\.\w+$','\.wav');
  [data, fs, nbits] = wavread(wav_file);
  data = data(:,1);
  data = data - mean(data);
end
%}}}

% Time preserving variables
feat_csum = cumsum(feat,2);            % LF-by-LT, cummalative sum of feat
feat2 = feat.^2;                       % LF-by-LT, squared feat
feat_len2 = sum(feat2,1);              %  1-by-LT, squared length of feat
feat_len2_csum  = cumsum(feat_len2,2); %  1-by-LT, cummulative sum of feat_len2
feattimesT = feat * diag(1:LT);
feattimesT_csum = cumsum(feattimesT,2);
feat_var = var(feat,0,2);
feat_std = sqrt(feat_var);
feat_mean = mean(feat,2);

% Record all possible segments, total will be (2*LT-1) after HAC
Segments = struct('start_t', 1:2*LT-1, ...
                  'end_t', 1:2*LT-1, ...
                  'var_sum', zeros(2*LT-1,1), ...
                  'parent', zeros(2*LT-1,1), ...
                  'child', zeros(2*LT-1,2), ...
                  'linerr', zeros(2*LT-1,1), ...
                  'slope', zeros(2*LT-1,LF), ...
                  'mergeloss', [], ...
                  'std_merge', 0);
num_seg = LT;
% Record merging candidate [index1, index2, mergeloss, var_sum, linregerr];
neighbor_frame_sqrsum   = filter([1 1], 1, feat_len2);
neighbor_frame_sqrsum(1)= [];
neighbor_frame_mean     = filter2([0.5 0.5], feat);
neighbor_frame_mean_sqr = sum(neighbor_frame_mean(:,1:end-1).^2,1);
%---------------------- Version 1 & 2 --------------------------
list = num2cell(1:6);
[L_SEG, R_SEG, LOSS, VARSUM, LINERR, SLOPE] = list{:};
%[L_SEG, R_SEG, LOSS]
Merge_cand = [(1:LT-1)', (2:LT)', (neighbor_frame_sqrsum - 2 * neighbor_frame_mean_sqr)'];
%[+ VARSUM, LINERR, SLOPE...]
Merge_cand = [Merge_cand, Merge_cand(:,3), zeros(LT-1,1), zeros(LT-1,LF)];
%---------------------------------------------------------------
Merge_cand = sortrows(Merge_cand,[-LOSS,-L_SEG]);
num_merge_left = LT-1;
Tree = zeros(num_merge_left,3);

max_loss = 0;
while( num_merge_left > 0 )
  % Create new segment
  % Merge_cand(1:num_merge_left,:) -- candidate to be merged (decreasing merge-loss)
  % Merge_cand(num_merge_left+1:end,:) -- Merged segments (first merge last)
  seg1 = Merge_cand(num_merge_left,L_SEG);
  seg2 = Merge_cand(num_merge_left,R_SEG);
  num_seg = num_seg + 1;
  Segments.start_t(num_seg) = Segments.start_t(seg1);
  Segments.end_t(num_seg)   = Segments.end_t(seg2);
  Segments.var_sum(num_seg) = Merge_cand(num_merge_left,VARSUM);
  Segments.parent([seg1,seg2]) = num_seg;
  Segments.child(num_seg,:) = [seg1, seg2];
  Segments.linerr(num_seg)  = Merge_cand(num_merge_left, LINERR);
  Segments.slope(num_seg,:)   = Merge_cand(num_merge_left, SLOPE : end);
  max_loss = max(max_loss,Merge_cand(num_merge_left,LOSS));
  Tree(end-num_merge_left+1,:) = [seg1, seg2, max_loss];

  %TODO!! fixme: if merging over three frames, calculate the total residue:)
  % Modify 3 candidates --> 2 candidates
  for i=1:2
    if (i==1)
      Imerge = find(Merge_cand(1:num_merge_left,R_SEG) == seg1);
    else
      Imerge = find(Merge_cand(1:num_merge_left,L_SEG) == seg2);
    end
    if (isempty(Imerge))
      continue;
    end
    if (i==1)
      n_seg1 = Merge_cand(Imerge,L_SEG);
      n_seg2 = num_seg;
      Merge_cand(Imerge,R_SEG) = n_seg2;
    else
      n_seg1 = num_seg;
      n_seg2 = Merge_cand(Imerge,R_SEG);
      Merge_cand(Imerge,L_SEG) = n_seg1;
    end

    start_t = Segments.start_t(n_seg1);
    end_t   = Segments.end_t(n_seg2);
    num_samp = end_t - start_t + 1;
    %% Calculate total variance
    sum_len2 = feat_len2_csum(end_t);
    m_vec = feat_csum(:,end_t);
    if (start_t ~= 1)
      sum_len2 = sum_len2 - feat_len2_csum(start_t-1);
      m_vec = m_vec - feat_csum(:,start_t-1);
    end
    new_var_sum = sum_len2  - (m_vec' * m_vec) / num_samp;
    Merge_cand(Imerge,VARSUM) = new_var_sum;
    %% Calculate linear regression error
    %%   Given feat(:,s:e) and time = s:e, we try to minimize the length of
    %%   feat(:,s:e) - coef * A,
    %%   where coef is a LF-by-2 coefficient matrix, A = [ (s : e); ones(1, e-s+1)].
    %%   The optimal solution is coef = (feat(:,s:e) * A') * inv(A * A')
    %%   Let FAt := feat(:,s:e) * A' (size=LFx2); AAt := A * A' (size=2x2).
    %%   We have coef = FAt * inv(AAt). Note that FAt and AAt can both be calculated in constant time
    %%   (not a function of (e-s+1)
    AAt = [1/6*(end_t * (end_t+1) * (2*end_t+1) - (start_t-1) * start_t * (2*start_t-1)), ...
           (start_t + end_t) * num_samp / 2; ...
           0, ...
           num_samp];
    AAt(2,1) = AAt(1,2);
    FAt = [feattimesT_csum(:,end_t), feat_csum(:,end_t)];
    if (start_t ~= 1)
      FAt = FAt - [feattimesT_csum(:,start_t-1), feat_csum(:,start_t-1)];
    end
    coef = FAt * inv(AAt);
    Merge_cand(Imerge, SLOPE:end) = coef(:,1)' ./ feat_var';
    %%   Now we solve coef, we need to know the total squared length of [feat(:,s:e) - coef * A]
    %%   Equivalently we can calculate:
    %%      sum_{f=1:LF} (feat(f,s:e) - coef(f,:) * A) * (feat(f,s:e) - coef(f,:) * A)'
    %%    = sum_{f=1:LF} [feat(f,s:e) * feat(f,s:e)'] + [coef(f,:) * A * A' * coef'] - 2 * [feat(f,s:e)' * A' * coef(f,:)']
    %%    = sum_{f=1:LF} [sum(feat2(f,s:e))][1] + [coef(f,:) * AAt * coef(f,:)'][2] - 2 * [feat(f,s:e)' * A' * coef(f,:)'][3]
    %%   [1] = (feat_len2_csum(e) - feat_len2_csum(s-1))
    %%   [2] = sum_{f=1:LF} coef(f,:) * AAt * coef(f,:)'
    %%   [3] = totsum([feattimesT_csum(s~e), feat_csum(s~e)] .* coef)
    toterr = feat_len2_csum(end_t);
    if (start_t ~= 1)
      toterr = toterr - feat_len2_csum(start_t-1);
    end
    for f = 1 : LF
      toterr = toterr + coef(f,:) * AAt * coef(f,:)';
    end
    toterr = toterr - 2 * sum(sum(FAt.*coef));
    Merge_cand(Imerge,LINERR) = toterr;

    %---------------------- Version 1 & 2 --------------------------
    if type == 1
      Merge_cand(Imerge,LOSS) = new_var_sum - Segments.var_sum(n_seg1) - Segments.var_sum(n_seg2);
    else
      Merge_cand(Imerge,LOSS) = toterr - Segments.linerr(n_seg1) - Segments.linerr(n_seg2);
    end
    %Merge_cand(Imerge,LOSS) = ( new_var_sum - Segments.var_sum(n_seg1) - Segments.var_sum(n_seg2) ) / num_samp;
    %---------------------------------------------------------------
  end
  num_merge_left = num_merge_left - 1;
  Merge_cand(1:num_merge_left,:) = sortrows(Merge_cand(1:num_merge_left,:),-3);
end

% Remove those merge loss that are too far from mean
mean_merge = mean(Tree(:,3));
std_merge  = std(Tree(:,3));
eff_L = 0;
while (Tree(end-eff_L,3) > mean_merge + 10 * std_merge)
  eff_L = eff_L + 1;
end
eff_merge  = Tree(1:end-eff_L,3);
% Calculate mean and std of merge-loss
mean_merge = mean(eff_merge);
std_merge  = std(eff_merge);
% Set seg_thresh to be ...
seg_thresh = mean_merge + threshold_parm1 * std_merge;
% Calculate the basic segments
merge_end = 0;
Num_of_seg = length(Segments.start_t);
Is_seg = true(1,Num_of_seg);
% Kill all segments that have been merged
while (Tree(merge_end+1,3) <= seg_thresh)
  merge_end = merge_end + 1;
  Is_seg(Tree(merge_end,1)) = false;
  Is_seg(Tree(merge_end,2)) = false;
end
% Kill all segments that have not been generate
Is_seg(LT + merge_end + 1:end) = false(1,Num_of_seg-LT-merge_end);
Index = 1:Num_of_seg;
boundary_idx = [winsize/2 + dT * ( Segments.start_t(Is_seg) - 1.5 );Index(Is_seg)];
boundary_idx = sortrows(boundary_idx',1)';
boundary = boundary_idx(1,:);
basic_seg_idx = boundary_idx(2,:);
boundary(1) = 0;
boundary = [boundary (LT-1)*dT+winsize];

boundary2 = ones(2,1) * boundary;

Segments.mergeloss = Tree(:,3);
Segments.std_merge = std_merge;


LF = size(original_feat, 1);
% Plotting preparation%{{{
% Calculate spectrogram dynamic range%{{{
if (~isdemo)
  return;
else
  feat_stat = reshape(original_feat,1,LT*LF);
  %spec_max = max(original_feat_stat);
  %spec_min = min(original_feat_stat);
  %spec_rang = spec_max - spec_min;
  spec_mean = mean(feat_stat);
  spec_std = std(feat_stat);
  thresh = sqrt(10)*spec_std;

  %subplot(2,1,1);
  %hold on;
  %hist(original_feat_stat,floor(length(original_feat_stat)/30));
  %hold off;

  feat_stat = feat_stat(abs(feat_stat - spec_mean) < thresh);
  spec_mean = mean(feat_stat);
  spec_std = std(feat_stat);
  %subplot(2,1,2);
  %hold on;
  %hist(feat_stat,floor(length(feat_stat)/30));
  %hold off;
end
%}}}
t_start = 0;
NumPlot = 2;


clf;
pos = {[0.05 0.4 0.9 0.6], [0.05 0.1 0.9 0.3]};
% Plot cluster tree on fig.1%{{{
ax1 = subplot('position',pos{1});
Tree(:,3) = log(1+Tree(:,3));
log_seg_thresh = log(1 + seg_thresh);
dendrogram(Tree,0,'colorthreshold',log_seg_thresh,'reorder',1:LT);
hold on;
plot([0 LT],log_seg_thresh*ones(1,2),'k','linewidth', 3);
hold off;
ylabel('log(L(i))','fontsize',14);
set(ax1,'xtick',[],'tickdir','out');
%}}}

% Plot spectrogram on fig.2%{{{
ax2 = subplot('position',pos{2});
hold on;
imagesc(T,F,original_feat,spec_mean+1.5*[-spec_std spec_std]);
if (type == 1)
  plot(boundary2,[F(1)-dF/2 F(end)+dF/2],'k');
elseif (type == 2)
  for i = 1:length(basic_seg_idx)
    %if (Segments.slope(basic_seg_idx(i)) < 10 * 0.01)
    %  continue;
    %end
    %num_frames_in_seg = Segments.end_t(basic_seg_idx(i)) - Segments.start_t(basic_seg_idx(i));
    %str_slope = sprintf('%.1f\n%.2f', 100 * Segments.slope(basic_seg_idx(i)), ...
    %                    Segments.linerr(basic_seg_idx(i))/num_frames_in_seg);
    %text(mean(boundary(i:i+1)), -F(2), str_slope, 'horizontalalignment', 'center','verticalalignment','top');
  end
  hold on;
  t_e = Segments.end_t(basic_seg_idx(1));
  right_seg_mean = feat_csum(:, t_e) / t_e;
  right_slope = Segments.slope(basic_seg_idx(1),:);
  for i = 2 : length(basic_seg_idx)
    % TODO: If rise & fall inside a segment => DTW
    % left/right_seg_mean
    left_seg_mean = right_seg_mean;
    t_s = Segments.start_t(basic_seg_idx(i));
    t_e = Segments.end_t(basic_seg_idx(i));
    right_seg_mean = ...
      (feat_csum(:, t_e) - feat_csum(:, t_s - 1)) / (t_e - t_s + 1);
    % left/right_slope
    left_slope = right_slope;
    right_slope = Segments.slope(basic_seg_idx(i),:);
    weighted_slope = (right_slope - left_slope) .* log(1+(exp(left_seg_mean) + exp(right_seg_mean)))';
    % left_bnd_feat
    t_w = Segments.start_t(basic_seg_idx(i - 1));
    if (t_s - t_w < LRwin)
      w = linspace(0, 1, t_s - t_w);
    else
      t_w = t_s - LRwin;
      w = linspace(0, 1, LRwin);
    end
    left_bnd_feat = feat(:, t_w:t_s-1) * w' / sum(w);
    % right_bnd_feat
    if (t_e - t_s < LRwin - 1)
      t_w = t_e;
      w = linspace(1, 0.1, t_e - t_s + 1);
    else
      t_w = t_s + LRwin - 1;
      w = linspace(1, 0.1, LRwin);
    end
    right_bnd_feat = feat(:, t_s:t_w) * w' / sum(w);
    diff_feat_normed = (feat(:,t_s) - feat(:,t_s-1)) ./ feat_std;
    % check whether a boundary
    if (sum(weighted_slope) >= 0) % A valley
      plot(boundary(i) * ones(1,2), [F(1)-dF/2 F(end)+dF/2], '-k', ...
           'linewidth', 4);
    elseif (abs(sum(weighted_slope)) / sum(abs(weighted_slope)) < 0.5) % rise&fall 
      plot(boundary(i) * ones(1,2), [F(1)-dF/2 F(end)+dF/2], '-k', ...
           'linewidth', 4);
    else
      plot(boundary(i) * ones(1,2), [F(1)-dF/2 F(end)+dF/2], '-k', ...
           'linewidth', 3);
    end
    %str_bnd = [];
    %weighted_slope_ = 100 * fliplr(weighted_slope);
    %for b = 1:nband
    %  str_bnd = sprintf('%s%.0f\n', str_bnd, weighted_slope_(b));
    %end
    %str_bnd = sprintf('%.1f\n%.1f\n%.1f\n%.1f', flipud(diff_feat_normed));
    %text(boundary(i), -F(2), str_bnd, 'horizontalalignment', 'center', ...
    %     'verticalalignment','top');
    %left_slope_ = 100 * fliplr(left_slope);
    %str_seg = [];
    %for b = 1:nband
    %  str_seg = sprintf('%s%.0f\n', str_seg, left_slope_(b));
    %end

    %if (abs(sum(left_slope)) / sum(abs(left_slope)) < 0.7)
    %  if (sum(abs(left_slope)) > 0.08)
    %    text(mean(boundary(i-1:i)), -F(2), str_seg, 'horizontalalignment', 'center', ...
    %         'verticalalignment', 'top', 'color', 'r');
    %  else
    %    text(mean(boundary(i-1:i)), -F(2), str_seg, 'horizontalalignment', 'center', ...
    %         'verticalalignment', 'top', 'color', 'y');
    %  end
    %else
    %  text(mean(boundary(i-1:i)), -F(2), str_seg, 'horizontalalignment', 'center', ...
    %       'verticalalignment', 'top', 'color', 'g');
    %end
  end
  hold off;
end
hold off;
set(gca,'ydir','normal','ytick',[], 'xtick', []);
%title(filename);
ylabel('Freq(Hz)','fontsize',14);
axis([T(1)-dT/2 T(end)+dT/2 F(1)-dF/2 F(end)+dF/2]);
%}}}

pos1 = get(ax1,'position');
pos2 = get(ax2,'position');
pos1(4) = pos1(2)+pos1(4)-pos2(2)-pos2(4);
pos1(2) = pos2(2)+pos2(4);
set(ax1,'position',pos1,'xtick',0:5:LT),
