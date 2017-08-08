import pandas as pd
import numpy as np

df_index = pd.read_csv('df_index.csv')

def get_resistance(df_index, window):
    df_index['ii'] = range(len(df_index))

    def window_resistance(ii, df):
        t_df = df.iloc[map(int, ii)]
        window_df = t_df.iloc[0: -1]
        window_df['dis_weight'] = window_df['ii']