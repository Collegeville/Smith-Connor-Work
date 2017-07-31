function matrices_to_file(matrix_list)

[m,n] = size(matrix_list);
max = n;

for i=1:max
    Problem = UFget(matrix_list(i));
    id = Problem.id;
    A = Problem.A;
    filename = sprintf('matrix%d', id)
    file = fopen(filename, 'w');
    [i,j,v] = find(A);
    nzmax = size(v,1);
    for ii=1:nzmax
        fprintf(file,'%9d %9d %21.17e\n', i(ii), j(ii), v(ii));
    end
end
