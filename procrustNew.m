function [A2, Q, r] = procrustNew( A, B )

% PROCRUST Orthogonal Procrustes problem
%    A2 = PROCRUST( A, B ) applies an orthogonal transformation to matrix B
%    by multiplication with Q such that A-B*Q has minimum Frobenius norm. The
%    results B*Q is returned as A2.
%
%    [A2, Q] = PROCRUST( A, B ) also returns the orthogonal matrix Q that was
%    used for the transformation.
%
%    [A2, Q, R] = PROCRUST( A, B ) also returns the Frobenius norm of A-B*Q.

% Author : E. Larsen
% Date   : 12/22/03
% Email  : erik.larsen@ieee.org

% Reference: Golub and van Loan, p. 601.

% Error checking
msg = nargchk( 2, 2, nargin );
if ~isempty( msg )
    error( msg )
end

% Do the computation
C = B.'*A;
[U1, S1, V1] = svd( C'*C );
[U2, S2, V2] = svd( C*C' );

Q = U2*V1';
A2 = B*Q;

% Optional output of norm
if nargout > 2
    r = norm( A-A2, 'fro' );
end
