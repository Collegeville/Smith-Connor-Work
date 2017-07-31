function [L] = generate_preconditioners(problem_array)

for i=1:size(problem_array)
    Problem = UFget(problem_array(i));
    A = Problem.A;
    [m,n] = size(A);
    if m < 50000
        L = chol(A,'lower');
    else 
        
        try
            L = ichol(A, struct('type','ict','droptol',1e-3, 'diagcomp', .01));
        catch NPP
            for n = .1:.1:1
            try 
                L = ichol(A, struct('type','ict','droptol',1e-3, 'diagcomp', n));
            catch
            end
            end
        end
    end
end
