function D=dotplot(M,N)    
% [m temp1]=size(M);
% [n temp2]=size(M);
% if ~temp1==temp2
%     message='error, dimentsions do not agree';
%     return
% end

%D=zeros(m,n);
D_dot=M*N';
M_abs=sum(M'.^2);%1-by-M vector
N_abs=sum(N'.^2);%1-by-N vector
D_sqrt=sqrt(M_abs'*N_abs);
D=0.5*(1+D_dot./D_sqrt);
