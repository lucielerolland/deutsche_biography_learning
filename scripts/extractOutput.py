import pandas as pd

url = '../data/'

def fullOutput(url):

	output = pd.read_csv(url+ 'output/output'+str(0)+'.csv')

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'output/output'+str(i)+'.csv')
		output = output.append(read, ignore_index=True)
		i=i+1

	return output

def orteOutput(url):
	output = pd.read_csv(url+ 'output/output'+str(0)+'.csv')
	output = output[output.iplace == 1]

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'output/output'+str(i)+'.csv')
		read = read[read.iplace == 1]
		output = output.append(read, ignore_index=True)
		i=i+1

	return output

def orteNoBioOutput(url):
	output = pd.read_csv(url+ 'output/output'+str(0)+'.csv')
	output = output[(output.iplace == 1) & (output.ibio == 0)]

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'output/output'+str(i)+'.csv')
		read = read[(read.iplace == 1) & (read.ibio == 0)]
		output = output.append(read, ignore_index=True)
		i=i+1

	return output

def orteBioOutput(url):
	output = pd.read_csv(url+ 'output/output'+str(0)+'.csv')
	output = output[(output.iplace == 1) & (output.ibio == 1)]

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'output/output'+str(i)+'.csv')
		read = read[(read.iplace == 1) & (read.ibio == 1)]
		output = output.append(read, ignore_index=True)
		i=i+1

	return output

def orteIntoPeopleDic(matrix, dic1={}):
	dic2 = dic1
	i=0
	while i < len(matrix):
		if matrix['idn'][i] != '':
			try:
				if 'orte' in dic2[matrix['idn'][i]].keys():
					dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
				else:
					dic2[matrix['idn'][i]]['orte'] = {}
					dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
			except KeyError:
				dic2[matrix['idn'][i]] = {}
				dic2[matrix['idn'][i]]['idn'] = matrix['idn'][i]
				dic2[matrix['idn'][i]]['name'] = matrix['name'][i]
				if 'orte' in dic2[matrix['idn'][i]].keys():
					dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
				else:
					dic2[matrix['idn'][i]]['orte'] = {}
					dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
		i=i+1

	return dic2

def locationList(matrix):
	loc = {}
	i=0
	while i < len(matrix):
		if matrix['cities'][i] != '' and matrix['cities'][i] not in loc.keys():
			loc[matrix['cities'][i]] = 1
		else:
			loc[matrix['cities'][i]] = loc[matrix['cities'][i]] + 1
		i=i+1

	return loc

#print(len(locationList(orteBioOutput(url))))

d=locationList(orteOutput(url))
dlist = []

for key, value in d.items():
    temp = [key,value]
    dlist.append(temp)

dlist.sort(key=lambda x:-x[1])

for i in range(0,99):
	print(dlist[i])

#print(len(locationList(orteNoBioOutput(url))))

#print(len(locationList(orteOutput(url))))






