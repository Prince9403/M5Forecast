import pandas as pd

def load_data():
    df_calendar = pd.read_csv("m5data/calendar.csv")
    df_sales = pd.read_csv("m5data/sales_train_validation.csv")

    return df_calendar, df_sales
