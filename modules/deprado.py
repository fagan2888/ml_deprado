#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook
import time
import datetime as dt
import sys


def getDailyVolatility(close, span0=100):
    """
    From de Prado - Daily Volatility
    Daily volatility reindexed to close - computes the daily volatility at
    intraday estimation points, applying a span of span0 days (bars) to an
    exponentially weighted moving standard deviation.

    # args
        close   : series of closing prices (could be from tick, volume or
                  dollar bars)
        span0   : number of days (or bars) to span
        bartype : type of bar being ohlc-d ("tick" <default>, "dollar",
                  "volume")
    # returns
        df0     : a dataframe with ohlc values
    """
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0-1],
                     index=close.index[close.shape[0]-df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index]/close.loc[df0.values].values-1
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')

    df0 = df0.ewm(span=span0).std().rename('dailyVolatility')

    return df0


def applyPtSlOnT1(close, events, ptSl, molecule):
    """
    From de Prado - apply stop loss / profit taking
    if it takes place before t1 (end of event)

    # args
        close    : a series of closing prices (could be from tick, volume or
                   dollar bars)
        events   : events
        ptSl     : profit-taking / stop loss multiples
        molecule : a list with the subset of event indices that will be
                   processed by a single thread.
    # returns
        df0     : a dataframe with ohlc values
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0]*events_['trgt']
    else:
        pt = pd.Series(index=events.index)

    if ptSl[1] > 0:
        sl = -ptSl[1]*events_['trgt']
    else:
        sl = pd.Series(index=events.index)
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0/close[loc]-1)*events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest SL
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest PT

    return out


def getEvents(close,
              tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    # args
        close:      A pandas series of prices.
        tEvents:    The pandas timeindex containing the timestamps
                    that will seed every triple barrier. These are the
                    timestamps selected by the sampling procedures discussed
                    in Chapter 2, Section 2.5.
        ptSl:       A non-negative float that sets the width of the two
                    barriers.
                    A 0 value means that the respective horizontal barrier
                    (profit taking and/or stop loss) will be disabled.
        t1:         A pandas series with the timestamps of the vertical
                    barriers. We pass a False when we want to disable vertical
                    barriers.
        trgt:       A pandas series of targets, expressed in terms of absolute
                    returns.
        minRet:     The minimum target return required for running a triple
                    barrier search.
        numThreads: The number of threads concurrently used by the function.
    # returns
        a pandas series of events
    """
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    # events= pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
    events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1)
              .dropna(subset=['trgt']))

    df0 = mpPandasObj(func=applyPtSlOnT1,
                      pdObj=('molecule', events.index),
                      numThreads=numThreads,
                      close=close,
                      events=events,
                      ptSl=ptSl_)

    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop('side', axis=1)
    return events


def getTEvents(gRaw, h):
    """
    From de Prado - Symmetrical CUSUM Filter

    The CUSUM filter is a quality-control method, designed to detect a shift
    in the mean value of a measured quantity away from a target value.

    # args
        gRaw : raw time series of closing prices (could be from tick, volume
               or dollar bars)
        h    : a threshold value
    # returns
        a series of index timestamps
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)

    return pd.DatetimeIndex(tEvents)


def addVerticalBarrier(tEvents, close, numDays=1):
    """
    From de Prado - add a vertical barrier, t1

    # args
        tEvents : threshold events
        close   : series of close prices
        numDays : number of days wide for the barrier
    # returns
        t1      : barrier timestamp
    """
    t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


def linParts(numAtoms, numThreads):
    """
    # partition of atoms with a single loop
    """
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms)+1)
    parts = np.ceil(parts).astype(int)
    return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    """
    # partition of atoms with an inner loop
    """
    parts, numThreads_ = [0], min(numThreads, numAtoms)

    for num in range(numThreads_):
        part = 1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part = (-1+part**.5)/2.
        parts.append(part)

    parts = np.round(parts).astype(int)

    if upperTriang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True,
                **kargs):
    """
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    """
    import pandas as pd
    # if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    # else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads*mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads*mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i-1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out

    for i in out:
        df0 = df0.append(i)

    df0 = df0.sort_index()
    return df0


def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp+' '+str(round(msg[0]*100, 2))+'% '+task+' done after ' \
                       + str(round(msg[1], 2))+' minutes. Remaining ' \
                       + str(round(msg[2], 2))+' minutes.'

    if jobNum < numJobs:
        sys.stderr.write(msg+'\r')
    else:
        sys.stderr.write(msg+'\n')
    return


def processJobs(jobs, task=None, numThreads=24):
    """
    Run in parallel.
    jobs must contain a 'func' callback, for expandCall
    """
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)

    outputs, out, time0 = pool.imap_unordered(expandCall,
                                              jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


def expandCall(kargs):
    """
    Expand the arguments of a callback function, kargs['func']
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def getBins(events, close):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


def dropLabels(events, minPct=.05):
    """
    # apply weights, drop labels with insufficient examples
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
    return events
