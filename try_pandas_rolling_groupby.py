import numpy as np
import pandas as pd

"""
df = pd.DataFrame(index=np.arange(5))

df['a'] = [1, 2, 5, 11, 21]
df['b'] = [2, 5, 8, 11, 20]

print(df)

# print(df['b'].rolling(2))

df['b'] = df['b'].rolling(3).sum()
print(df)
"""

df = pd.DataFrame(index=np.arange(5))

df['a'] = ['id', 'rt', 'id', 'id', 'rt']
df['b'] = [2, 5, 8, 11, 20]
print(df)

print(df.groupby(['a'])['b'].sum())

# df['b'] = df.groupby(['a'])['b'].transform(lambda x: x.rolling(2).sum())
# print(df)

df['c'] = df.groupby(['a'])['b'].transform(lambda x: x.shift(1))
print(df)