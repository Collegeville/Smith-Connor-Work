function matrices_to_file(matrix_list)

[m,max] = size(matrix_list);

filename = ('matrix_data.csv');

file = fopen(filename, 'a');

%fprintf(file,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', 'Size', 'Nonzeros', 'Symmetry', '2D/3D', 'Pos/Def', 'Kind', 'Solver', 'Tol', 'Maxit', 'Precond', 'droptol', 'diagcomp');

for i=1:max
    Problem = UFget(matrix_list(i));
    
    id = Problem.id;
    
    A = Problem.A;
    
    m = sprank(A);
    
    nonzeros = nnz(A);
    
    symmetry = issymmetric(A);
    
    discretization = 1;
    
    posdef = 1;
    
    kind = Problem.kind;
    
    solver = 'cg';
    
    tol = 1e-6;
    
    maxit = 100;
    
    precond = 'ichol';
    
    droptol = 0;
    
    diagcomp = 0;

    fprintf(file,'%9d,%9d,%9d, %9d, %9d, %s, %s, %9e, %9d, %s, %9e, %9d\n', m, nonzeros, symmetry, discretization, posdef, kind, solver, tol, maxit, precond, droptol, diagcomp);
end
fclose(file);
