import pickle 
import numpy as np

symm1 = np.zeros(16)
symm1[0] = 1
symm1[5] = 1
symm1[10] = 1
symm1[15] = 1

symm2 = np.zeros(25)
symm2[0] = 1
symm2[6] = 1
symm2[12] = 1
symm2[18] = 1
symm2[24] = 1

nonSymm1 = np.zeros(16)
nonSymm1[5] = 1
nonSymm1[6] = 1
nonSymm1[15] = 1

matrices = [symm1, symm2, nonSymm1]

file_name = "testfile"

fileObject = open(file_name, 'wb')

pickle.dump(matrices, fileObject)

fileObject.close()

fileObject = open(file_name,'rb')  

b = pickle.load(fileObject)  

fileObject.close()

print(b)

