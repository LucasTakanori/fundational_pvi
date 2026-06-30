function [x, fval] = compute_emd(S1, S2)
    arguments
        S1 {mustBeNumeric}
        S2 {mustBeNumeric}
    end

    M = size(S1, 1);
    N = size(S2, 1);

    norms_A = sum(S1.^2, 2);
    norms_B = sum(S2.^2, 2);

    C = sqrt(norms_B + norms_A' - 2*(S2*S1'));
    c = C(:);

    % A1 = kron(eye(M), ones(1, N));

    % A1 = zeros(M, M*N);
    
    A2 = sparse(diag(ones(1,N)));
    A2 = repmat(A2,1,M);
    
    A = [A1; A2];
    b = [ones(M,1)/M; ones(N,1)/N];

    Aeq = ones(M + N, M * N);
    beq = ones(M + N, 1);
    
    lb = zeros(1, M * N);

    [x, fval] = linprog(c, A, b, Aeq, beq, lb);
    fval = fval / sum(x);
end