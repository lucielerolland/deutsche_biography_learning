import pandas
import extractInput as ei
import extractOutput as eo



#def

#dic = {}

dic = ei.bioIntoPeopleDic(ei.bioOrteInput(url), dic)

dic = eo.orteIntoPeopleDic(eo.orteBioOutput(url), dic)

#print(dic['118613960'])
