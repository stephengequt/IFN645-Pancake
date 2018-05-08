import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#Selecting and retrieving data
series_obj = Series(np.arange(8), index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6', 'row 7', 'row 8'])

series_obj
series_obj['row 7']
series_obj[[0,4]]

np.random.seed(5)
DF_obj = DataFrame(np.random.rand(36).reshape((6,6)), index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6'],
                   columns=['column1', 'column2', 'column3', 'column4', 'column5', 'column6'])
DF_obj

DF_obj.iloc[[0, 2], DF_obj.columns.get_indexer(['column2', 'column3'])]