#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


def get_ma_cross_up(df):
    """
    Find a MA Up Cross
    """
    crit1 = df.fast.shift(1) < df.slow.shift(1)
    crit2 = df.fast > df.slow
    return pd.DataFrame(df.fast[(crit1) & (crit2)])


def get_ma_cross_down(df):
    """
    Find a MA Down Cross
    """
    crit1 = df.fast.shift(1) > df.slow.shift(1)
    crit2 = df.fast < df.slow
    return pd.DataFrame(df.fast[(crit1) & (crit2)])


def rel_strength_indicator(close, n=14):
    """
    Calculate the RSI Indicator
    """
    delta = close.diff()
    delta = delta[1:]

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RollUp = dUp.rolling(n).mean()
    RolDown = dDown.rolling(n).mean().abs()

    RS = RollUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI


def get_up_rsi(df):
    """
    Find a RSI Up signal
    """
    crit1 = df.RSI > 30
    crit2 = df.RSI.shift(-1) <= 30
    return pd.DataFrame(df.RSI[(crit1) & (crit2)])


def get_down_rsi(df):
    """
    Find a RSI Down signal
    """
    crit1 = df.RSI < 70
    crit2 = df.RSI.shift(-1) >= 70
    return pd.DataFrame(df.RSI[(crit1) & (crit2)])


def get_aroon_cross_up(df):
    """
    Find an Aroon Up Cross
    """
    crit1 = df.AroonUp.shift(1) < df.AroonDn.shift(1)
    crit2 = df.AroonUp > df.AroonDn
    return pd.DataFrame(df.AroonUp[(crit1) & (crit2)])


def get_aroon_cross_down(df):
    """
    Find an Aroon Down Cross
    """
    crit1 = df.AroonUp.shift(1) > df.AroonDn.shift(1)
    crit2 = df.AroonUp < df.AroonDn
    return pd.DataFrame(df.AroonUp[(crit1) & (crit2)])


if __name__ == "__main__":
    main()
