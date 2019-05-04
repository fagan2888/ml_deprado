#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm, tqdm_notebook
import pandas as pd


def create_bars(df, col, m, bartype="tick"):
    '''
    Function to compute various types of bars, depending on column passed

    # args
        df: dataframe of tick price data
        p: name of column in df to calculate bars from
        m: int(), arbitrary threshold value for indicating bar size
        bartype : type of bar being ohlc'd ("tick" <default>, "dollar",
                  "volume")
    # returns
        idx: list of bar indices
    '''
    t = df[col]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        if bartype is "tick":
            ts += 1
        elif bartype is "volume" or bartype is "dollar":
            ts += x
            # print(x, ts)
        else:
            raise Exception('bartype must be one of "tick", "volume", \
                "dollar"')

        if ts >= m:
            idx.append(i)
            ts = 0
            continue

    return idx


def create_bar_df(df, col, m, bartype="tick"):
    '''
    Function to create a dataframe of custom bars, depending on column passed

    # args
        df: dataframe of original tick price data
        p: name of column in df to calculate bars from
        m: int(), somewhat arbitrary threshold value for indicating bar size
        bartype : type of bar being ohlc-d ("tick" <default>, "dollar",
                  "volume")
    # returns
        a dataframe of custom bars
    '''
    idx = create_bars(df, col, m, bartype)
    return df.iloc[idx].drop_duplicates()


# OHLC
def calc_ohlc(reference_col, subset_df, bartype="tick"):
    '''
    Function to create ohlc from custom bars

    # args
        reference_col : reference column from full tick data
        subset_df :  DataFrame that contains bars of our type
        bartype : type of bar being ohlc-d ("tick" <default>, "dollar",
                  "volume")
    # returns
        a dataframe with ohlc values
    '''

    # Note: the following allows us to add the closing price of the bar
    # to bars that are not based on price, which is essential for
    # calculating returns later.
    price_col = subset_df.Price

    if bartype is "tick":
        subset_col = subset_df.Price
    elif bartype is "volume":
        subset_col = subset_df.Volume
    elif bartype is "dollar":
        subset_col = subset_df.DollarVol
    else:
        raise Exception('bartype must be one of "tick", "volume", or "dollar"')

    ohlc = []
    for i in tqdm(range(subset_col.index.shape[0]-1)):
        start, end = subset_col.index[i], subset_col.index[i+1]
        # print(reference_col.loc[start:end])
        tmp_ref = reference_col.loc[start:end]
        end_price = price_col.iloc[i+1]
        max_px, min_px = tmp_ref.max(), tmp_ref.min()
        o, h, l, c = subset_col.iloc[i], max_px, min_px, subset_col.iloc[i+1]
        ohlc.append((end, start, o, h, l, c, price_col[i+1]))
    cols = ['End', 'Start', 'Open', 'High', 'Low', 'Close', 'ClosePrice']
    return (pd.DataFrame(ohlc, columns=cols))
