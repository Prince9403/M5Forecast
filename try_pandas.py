import numpy as np
import pandas as pd


df_4 = pd.DataFrame(index=np.arange(3), columns=['a', 'b'])
df_4['a'] = [2, 3, 8]
df_4['b'] = [16, 22, 47]
print(df_4.columns)
df_4['b'] = df_4['b'].transform(lambda x : x.shift(1))
print(df_4)


