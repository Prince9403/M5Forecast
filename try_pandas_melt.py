import numpy as np
import pandas as pd

a = pd.DataFrame(index = np.arange(3), columns=['id', 'store', 'item', 'sales'])
a['id'] = ['i14', 'i16', 'i25']
a['store'] = ['s2', 's8', 's10']
a['item'] = ['a7', 'a11', 'a22']
a['sales'] = [0.1, 17.8, 223.46]
print(a)

# b = pd.melt(a, id_vars=['id', 'store', 'item'])
# print(b)

c = pd.melt(a, id_vars=['id', 'store'])
print(c)