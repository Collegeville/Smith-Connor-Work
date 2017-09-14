#This program converts the columns with strings in a .csv file to 
#integers in order to read the file through a Neural Network

import csv

string_file = 'shuffled_new.csv'

####################################################################
##Convert all strings to int in a specified column within csv file##
####################################################################
def to_int(col):
	with open(string_file, 'r') as arraycsv:
		csvreader = csv.reader(arraycsv)
		count = 0
		string_list = list()
		for row in csvreader:
			if count != 0:
				if row[col] not in string_list:
					string_list.append(row[col])
			count += 1
		
	with open(string_file, 'r') as intcsv:
		csvreader = csv.reader(intcsv)
		intvals = list()
		count = 0
		for row in csvreader:
			if count != 0:
				thisstring = row[col]
				thisindex = string_list.index(thisstring)
				intvals.append(thisindex)
			count += 1

	return intvals, string_list
####################################################################


####################################################
##Encode strings to use in training and predicting##
####################################################

def encode(col, string_val):
	intvals, string_list = to_int(col)
	return string_list.index(string_val)
####################################################


#####################################
##Decode int values back to strings##
#####################################
def decode(col, index):
	intvals, string_list = to_int(col)
	return string_list[index]
#####################################


#####################################################
##Create lists of columns to print out into new csv##
#####################################################
def main():
	with open(string_file,'r') as stringcsv:
		csvreader = csv.reader(stringcsv)
		count = 0
		col0 = list()
		col1 = list()
		col2 = list()
		col3 = list()
		col4 = list()
		col5, string_list = to_int(5)
		col6, string_list = to_int(6)
		col7 = list()
		col8 = list()
		col9, string_list = to_int(9)
		col10 = list()
		col11 = list()
		col12 = list()
		col13 = list()
		for row in csvreader:
			if count != 0:
				col0.append(float(row[0]))
				col1.append(float(row[1]))
				col2.append(row[2])
				col3.append(row[3])
				col4.append(row[4])
				col7.append(float(row[7]))
				col8.append(row[8])
				col10.append(row[10])
				col11.append(row[11])
				col12.append(row[12])
				col13.append(row[13])
			count += 1
#SCALE DATA HERE???
#####################################################


	###########################################
	##Print encoded columns into new csv file##
	###########################################
	int_file = 'encoded_new.csv'
	with open(int_file,'w') as intcsv:
		csvwriter = csv.writer(intcsv, delimiter=',', lineterminator='\n')

		for i in range(0,len(col0)):
			csvwriter.writerow([col0[i], col1[i], col2[i], col3[i], col4[i], col5[i], col6[i], col7[i], col8[i], col9[i], col10[i], col11[i], col12[i], col13[i]])
	###########################################
	
	



