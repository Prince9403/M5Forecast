import sys
import time

import numpy as np
import pandas as pd

make_exp_smoothing = False
choose_alpha_for_exp_smoothing = True

start_time = time.time()

df_calendar = pd.read_csv("calendar.csv")

print(df_calendar.head(5))

df_sales = pd.read_csv("sales_train_validation.csv")

print(df_sales.head(5))

# print(df_sales.describe())

print("Number of sales series:", len(df_sales))

sys.exit(54)

print("Different stores:", set(df_sales['store_id']))

output_rows = list(df_sales['id'])
output_rows_eval = [item_id[:-10] + 'evaluation' for item_id in output_rows]
# print( output_rows_eval[:15])
output_rows = output_rows + output_rows_eval

df_out = pd.DataFrame(index = np.arange(len(output_rows)), columns = ['id'] + [f'F{i}' for i in range(1, 29)])
df_out['id'] = output_rows


# prediction by exponential smoothing
if make_exp_smoothing:
    alpha = 0.2

    for i in range(len(df_sales)):
        sales_series = df_sales.iloc[i, 6:].values
        n = len(sales_series)
        S = sales_series[0]
        for j in range(n):
            S = (1-alpha) * S + alpha * sales_series[j]
        out = S * np.ones(28)
        df_out.iloc[i, 1:] = out
        df_out.iloc[i + len(df_sales), 1:] = out

    df_out.to_csv("output_exp_smoothing_alpha_02.csv", index=False)

if choose_alpha_for_exp_smoothing:
    alphas_list = [0.07, 0.1, 0.15, 0.2, 0.3, 0.5]

    for i in range(len(df_sales)):
        sales_series = df_sales.iloc[i, 6:].values
        n = len(sales_series)

        min_S = 0

        for k in range(len(alphas_list)):
            alpha = alphas_list[k]
            S = sales_series[n - 100]
            total_error = 0
            for j in range(n-100, n):
                S = (1 - alpha) * S + alpha * sales_series[j]
                if j >= n - 15:
                    total_error += (S-sales_series[j]) ** 2
            if k == 0:
                min_error = total_error
                min_S = S
            else:
                if total_error < min_error:
                    min_error = total_error
                    min_S = S

        out = min_S * np.ones(28)
        df_out.iloc[i, 1:] = out
        df_out.iloc[i + len(df_sales), 1:] = out

     # df_out.to_csv("output_exp_smoothing_choose_alpha_last_15.csv", index=False)


end_time = time.time()
seconds = end_time - start_time
print(f"Program took {seconds/60} minutes")
