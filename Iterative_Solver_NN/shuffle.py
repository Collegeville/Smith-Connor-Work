import random
fid = open("encoded.csv", "r")
first = fid.readline()
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("shuffled_data.csv", "w")
fid.writelines(first)
fid.writelines(li)
fid.close()
