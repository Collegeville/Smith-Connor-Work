import csv


DATA_FILE = "data.csv"

with open(DATA_FILE,'r') as oldcsv:
		csvreader = csv.reader(oldcsv)
		count = 0
		col0 = list()
		col1 = list()
		col2 = list()
		col3 = list()
		col4 = list()
		col5 = list()
		col6 = list()
		col7 = list()
		col8 = list()
		col9 = list()
		col10 = list()
		col11 = list()
		col12 = list()
		col13 = list()
		col14 = list()
		col15 = list()
		col16 = list()
		col17 = list()
		col18 = list()
		col19 = list()
		col20 = list()
		col21 = list()
		col22 = list()
		col23 = list()
		col24 = list()
		col25 = list()
		col26 = list()
		col27 = list()
		col28 = list()
		col29 = list()
		col30 = list()
		for row in csvreader:
				col0.append(row[0])
				col1.append(row[1])
				col2.append(row[2])
				col3.append(row[3])
				col4.append(row[4])
				col5.append(row[5])
				col6.append(row[6])
				col7.append(row[7])
				col8.append(row[8])
				col9.append(row[9])
				col10.append(row[10])
				col11.append(row[11])
				col12.append(row[12])
				col13.append(row[13])
				col14.append(row[14])
				col15.append(row[15])
				col16.append(row[16])
				col17.append(row[17])
				col18.append(row[18])
				col19.append(row[19])
				col20.append(row[20])
				col21.append(row[21])
				col22.append(row[22])
				col23.append(row[23])
				col24.append(row[24])
				col25.append(row[25])
				col26.append(row[26])
				col27.append(row[27])
				col28.append(row[28])
				col29.append(row[29])
				if count > 0:
					if (int(row[30]) == -1):
						col30.append(0)
					elif (int(row[30]) == 0):
						col30.append(1)
					else:
						col30.append(2)
				else:
					col30.append(row[30])
				count += 1

			

new_file = 'new_data.csv'



with open(new_file,'w+') as newcsv:
	csvwriter = csv.writer(newcsv, delimiter=',', lineterminator='\n')
	
	
	for i in range(0,len(col0)):
		csvwriter.writerow([col0[i], col1[i], col2[i], col3[i], col4[i], col5[i], col6[i], col7[i], col8[i], col9[i], col10[i], col11[i], col12[i], col13[i], col14[i], col15[i], col16[i], col17[i], col18[i], col19[i], col20[i], col21[i], col22[i], col23[i], col24[i], col25[i], col26[i], col27[i], col28[i], col29[i], col30[i]])

	
