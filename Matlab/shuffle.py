import random
fid = open("matrix_data.csv", "r")
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("shuffled_data.csv", "w")
fid.writelines(li)
fid.close()
