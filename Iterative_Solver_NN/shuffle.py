import random
fid = open("matrix_data.csv", "r")
first = fid.readline()
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("shuffled_new.csv", "w")
fid.writelines(first)
fid.writelines(li)
fid.close()
