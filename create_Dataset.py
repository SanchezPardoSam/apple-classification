# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import mahotas
import os
import cv2
from PIL import Image, ImageStat


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def createDataset(df,name):
    try:
        salidacsv = open(name+'.csv', 'w')
        print('Se ha creado el archivo "' + name + '.csv"')

    finally:
        salidacsv.close()
    df.to_csv(name+'.csv')
    
def valoresHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(hsv, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    
    return avg_color[0], avg_color[1], avg_color[2]
        
def textures(img):
     return mahotas.features.haralick(img).mean(axis=0)
 
def brillo(img):
    im = Image.fromarray(img)    
    im.convert('L')
    stat = ImageStat.Stat(im)
    return stat.rms[0]

#def perimetro(gray):
#    ret,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
#    contornos,jerarquia = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#    cnt=contornos[0]    
        
#    return cv2.arcLength(cnt,True)
        
etiq = np.array(['Apple Braeburn','Apple Golden 3','Apple Granny Smith','Apple Pink Lady','Apple Red Delicious'])
pos = np.array([1,2,3,4,5])

datos = {'Matiz': [], 'Saturacion': [], 'Valor': [] ,'Brillo': [],'SRE': [], 'LRE':[], 'GNL':[], 'RLNU':[], 'RP':[],
         'LGRE':[], 'HGRE':[], 'SRLGE':[], 'SRHGE':[], 'LRLGE':[], 'LRHGE':[], 'GLV':[], 'RLV':[],'Etiqueta': []}
 
 # Images paths
path_originals = '.\\Training\\'
con = 0
#path_edited='/content/drive/MyDrive/DataSet/flores/jpg-2'
for x in range(len(etiq)):
    
    for i in os.listdir(path_originals + etiq[x]):
        
        img = cv2.imread(os.path.join(path_originals + etiq[x] , i))        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           

        SRE, LRE, GNL, RLNU, RP, LGRE, HGRE, SRLGE, SRHGE, LRLGE, LRHGE, GLV, RLV = textures(img)
        
        h, s, v = valoresHSV(img)
        
        brill = brillo(img)
        
        #peri = perimetro(gray)       
        #if peri <=100:
            #print(i)
        #    print(peri)
        #    con +=1
               
        datos['Matiz'].append(h)
        datos['Saturacion'].append(s)
        datos['Valor'].append(v)
        
        datos['Brillo'].append(brill)
        #datos['Perimetro'].append(peri)
        
        datos['SRE'].append(SRE)
        datos['LRE'].append(LRE)
        datos['GNL'].append(GNL)
        datos['RLNU'].append(RLNU)
        
        datos['RP'].append(RP)
        datos['LGRE'].append(LGRE)
        datos['HGRE'].append(HGRE)
        datos['SRLGE'].append(SRLGE)
        
        datos['SRHGE'].append(SRHGE)
        datos['LRLGE'].append(LRLGE)
        datos['LRHGE'].append(LRHGE)
        datos['GLV'].append(GLV)
        datos['RLV'].append(RLV)
        
        
        datos['Etiqueta'].append(pos[x])


print(con)
df  = pd.DataFrame(datos)
print(df)

train, validate, test = train_validate_test_split(df)
#del(train['Unnamed: 0'])

del(validate['Etiqueta'])
del(test['Etiqueta'])

createDataset(df,'dataset')
createDataset(train,'train')
createDataset(validate,'validate')
createDataset(test,'test')

