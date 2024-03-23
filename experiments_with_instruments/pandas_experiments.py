import numpy as np
import pandas as pd

x = pd.DataFrame(index=np.arange(3))

x['a'] = [2, 3, 22]
x['b'] = [23, 17, 64]
x['c'] = [2, 4, 8]
print(x)

y = pd.melt(x, id_vars=['b'])
print(y)