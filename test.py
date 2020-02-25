import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

df_train = pd.read_csv('dataset_n/labels.csv', sep=',', encoding='utf8')
df_test = pd.read_csv('dataset_n/labels_test.csv', sep=',', encoding='utf8')

print(df_train.head(10))

for i in df_train.columns: # перебираем все столбцы
    if str(df_train[i].dtype) == 'nan': # если тип столбца - object
        print('='*10)
        print(i) # выводим название столбца
        print(set(df_train[i])) # выводим все его значения (но делаем set - чтоб значения не повторялись)
        print('\n') # выводим пустую строку
