import datetime

import numpy as np
import pandas as pd
import pyodbc

def get_all_sales_by_articule_and_filial(connection, articule, filial, start_date, end_date):
    sql_query_sales_by_articule_and_filial = f"select [Date], \
            QtySales as quantity, \
            StoreQtyDefault as residue, \
            PriceOut as price, \
            MechanicId as promo_type \
            from [SalesHub.Dev].[DataHub].[SalesStores] s with (nolock, INDEX(pk_SalesStores)) \
            where LagerId = {articule}\
            and [Date] >= '{start_date}' and [Date] <= '{end_date}' \
            and FilialId = {filial}"
    df = pd.read_sql_query(sql_query_sales_by_articule_and_filial, connection)
    df = df.sort_values(by='Date')
    df.fillna({'quantity': 0}, inplace=True)
    df.fillna({'residue': 0}, inplace=True)
    df.fillna({'price': 0}, inplace=True)
    df.fillna({'promo_type': 0}, inplace=True)
    return df


def get_df_without_empty_dates(df, start_date, end_date):

    m = (end_date - start_date).days + 1

    old_dates = df['Date'].values
    old_quantities = df['quantity'].values
    old_prices = df['price'].values
    old_residues = df['residue'].values
    old_promo_types = df['promo_type'].values

    new_dates = np.array([start_date + datetime.timedelta(days=i) for i in range(m)])
    new_quantities = np.zeros(m)
    new_prices = np.zeros(m)
    new_residues = np.zeros(m)
    new_promo_types = np.zeros(m)

    i = 0
    for j in range(len(old_dates)):
        if old_dates[j] > end_date:
            break
        while new_dates[i] < old_dates[j]:
            i += 1
        if new_dates[i] == old_dates[j]:
            new_quantities[i] = old_quantities[j]
            new_prices[i] = old_prices[j]
            new_residues[i] = old_residues[j]
            new_promo_types[i] = old_promo_types[j]

    current_price = 0

    for i in range(m):
        if new_quantities[i] > 0:
            current_price = new_prices[i]
        else:
            new_prices[i] = current_price

    df_new = pd.DataFrame(columns=['Date', 'quantity', 'price', 'residue', 'is_special_day'], index=np.arange(m))

    df_new['Date'] = new_dates
    df_new['quantity'] = new_quantities
    df_new['price'] = new_prices
    df_new['residue'] = new_residues
    df_new['is_special_day'] = new_promo_types
    return df_new


connection = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                            "Server=S-KV-CENTER-S27;"
                            "Database=4t.Dev;"
                            "Trusted_Connection=yes;")

start_date = datetime.date(2017, 1, 8)
end_date = datetime.date(2019, 9, 30)

new_start = end_date + datetime.timedelta(days=1)
new_end = new_start + datetime.timedelta(days=30)

num_days_to_skip = 40

early_start = new_start - datetime.timedelta(days=num_days_to_skip)

articule = 32485
filial = 2016

df = get_all_sales_by_articule_and_filial(connection, articule, filial, early_start, new_end)

print(df.head(5))
print(df.tail(5))

print("***")

df = get_df_without_empty_dates(df, start_date, end_date)

print(df.head(5))
print(df.tail(5))

