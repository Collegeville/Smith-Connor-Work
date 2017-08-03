import csv

with open('shuffled_data.csv', 'r') as arraycsv:
	csvreader = csv.reader(arraycsv)
	count = 0
	string_list = list()
	for row in csvreader:
		if count != 0:
			if row[5] not in string_list:
				string_list.append(row[5])
		count += 1
		
with open('shuffled_data.csv', 'r') as intcsv:
	reader = csv.reader(intcsv)
	intvals = list()
	count = 0
	for row in reader:
		if count != 0:
			thisstring = row[5]
			thisindex = string_list.index(thisstring)
			intvals.append(thisindex)
		count += 1
	
	
	



