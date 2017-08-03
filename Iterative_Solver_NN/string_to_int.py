import csv

#cols 5,6,9

string_file = 'test_data.csv'

def to_int(col):
	with open(string_file, 'r') as arraycsv:
		csvreader = csv.reader(arraycsv)
		count = 0
		string_list = list()
		for row in csvreader:
			if count != 0:
				if row[5] not in string_list:
					string_list.append(row[5])
			count += 1
		
	with open(string_file, 'r') as intcsv:
		csvreader = csv.reader(intcsv)
		intvals = list()
		count = 0
		for row in csvreader:
			if count != 0:
				thisstring = row[5]
				thisindex = string_list.index(thisstring)
				intvals.append(thisindex)
			count += 1

	return intvals


def main():
	with open(string_file,'r') as stringcsv:
		csvreader = csv.reader(stringcsv)
		count = 0
		col0 = list()
		col1 = list()
		col2 = list()
		col3 = list()
		col4 = list()
		col7 = list()
		col8 = list()
		col10 = list()
		col11 = list()
		for row in csvreader:
			if count != 0:
				col0.append(row[0])
				col1.append(row[1])
				col2.append(row[2])
				col3.append(row[3])
				col4.append(row[4])
				col7.append(row[7])
				col8.append(row[8])
				col10.append(row[10])
				col11.append(row[11])

			count += 1



	int_file = 'encoded.csv'
	with open(int_file,'w') as intcsv:
		csvwriter = csv.writer(intcsv)
		i = 0
		col5 = to_int(5)
		col6 = to_int(6)
		col9 = to_int(9)
		while i < count:
			csvwriter.writerow([col0[i], col1[i], col2[i], col3[i], col4[i], col5[i], col6[i], col7[i], col8[i], col9[i], col10[i], col11[i]])
			i += 1

	
	
	



