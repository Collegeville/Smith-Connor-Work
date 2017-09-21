import random
fid = open("non_pos_def_data.csv", "r")
first = fid.readline()
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("shuffled_non_pos_def.csv", "w")
fid.writelines(first)
fid.writelines(li)
fid.close()
