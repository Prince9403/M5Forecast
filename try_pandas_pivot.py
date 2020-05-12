import numpy as np
import pandas as pd


df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                           'two'],
                   'bar': ['A', 'B', 'C', 'D', 'A', 'E'],
                   'baz': [1, 2, 3, 4, 5, 6],
                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})

print(df)

df_1 = df.pivot(index='foo', columns='bar', values='baz')

print(df_1)