import numpy as np
import pandas as pd 
import xlrd 

file_loc = 'MOESM1.xlsx'

df = pd.ExcelFile(file_loc).parse('Fluoride')
x = []
x.append(df[['Abstract','Included']])

Included = df[df['Included']=='INCLUDED']

Excluded = df[df['Included']=='EXCLUDED']

 


for row in Included.index:
    
    file_name = 'sample'+ str(Included['Refid'][row])
    

    text_file = open("Included\\"+file_name+'.txt', "w" , encoding= "utf-8" )
    
    n = text_file.write(str(Included['Abstract'][row]))
    text_file.close()
    
for row in Excluded.index:
    
    file_name = 'sample'+ str(Excluded['Refid'][row])
    

    text_file = open("Excluded\\"+file_name+'.txt', "w" , encoding= "utf-8" )
    
    n = text_file.write(str(Excluded['Abstract'][row]))
    text_file.close()
    
    
