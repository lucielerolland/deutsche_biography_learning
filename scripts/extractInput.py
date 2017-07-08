import pandas as pd

url = '/home/lucie/Bureau/2015-2016 LSE/Side work/RA/'

def fullInput(url):

	input = pd.read_csv(url+ 'input/input'+str(0)+'.csv')

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'input/input'+str(i)+'.csv')
		input = input.append(read, ignore_index=True)
		i=i+1

	return input

def bioInput(url):
	input = pd.read_csv(url+ 'input/input'+str(0)+'.csv')
	input = input[input.iplace == 1]

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'input/input'+str(i)+'.csv')
		read = read[read.ibio == 1]
		input = input.append(read, ignore_index=True)
		i=i+1

	return input

def bioNoOrteInput(url):
	input = pd.read_csv(url+ 'input/input'+str(0)+'.csv')
	input = input[(input.ibio == 1) & (input.iplace == 0)]

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'input/input'+str(i)+'.csv')
		read = read[(read.ibio == 1) & (read.iplace == 0)]
		input = input.append(read, ignore_index=True)
		i=i+1

	return input

def bioOrteInput(url):
	input = pd.read_csv(url+ 'input/input'+str(0)+'.csv')
	input = input[(input.ibio == 1) & (input.iplace == 1)]

	i=1
	while i < 148:
		read = pd.DataFrame({})
		read = pd.read_csv(url+ 'input/input'+str(i)+'.csv')
		read = read[(read.ibio == 1) & (read.iplace == 1)]
		input = input.append(read, ignore_index=True)
		i=i+1

	return input

def bioIntoPeopleDic(matrix, dic1={}):
	dic2 = dic1
	i=0
	while i < len(matrix):
		if matrix['idn'][i] != '':
			try:
				dic2[matrix['idn'][i]]['leben'] = matrix['leben'][i]
			except KeyError:
				dic2[matrix['idn'][i]] = {}
				dic2[matrix['idn'][i]]['idn'] = matrix['idn'][i]
				dic2[matrix['idn'][i]]['name'] = matrix['name'][i]
				dic2[matrix['idn'][i]]['leben'] = matrix['leben'][i]
		i=i+1
	return dic2
