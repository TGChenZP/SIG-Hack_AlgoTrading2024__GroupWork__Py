
import numpy as np
import pandas as pd
import statsmodels.api as sm

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
models = {}

tradedProducts = range(nInst)

def getFeature(df, nt, feature_name):
    arr = feature_name.split('_')
    stock_id = int(arr[1])
    window = int(arr[3])
    lookback_no = int(arr[5])
    return df.loc[nt - 1 - window * lookback_no , f'stock_{stock_id}'] - df.loc[nt - 1 - window * (lookback_no + 1), f'stock_{stock_id}']

def retrain(df_train):
    global models, nInst
    pd.options.mode.chained_assignment = None
    alpha = 0.01
    models = {}

    max_lags = 2
    target_forecast = 20
    windows = [5, 10, 20]
    rsquared_limit = 0.1

    for i in range(nInst):
        print('.', end = '')
        current_stock_signals = []
        ## FIND SIGNALS IN DATA:
        for j in range(nInst):
            for window in windows:
                for offset in range(0, max_lags):
                    df_temp = pd.DataFrame()
                    df_temp['target'] = df_train[f'stock_{i}'].diff(target_forecast)
                    df_temp[f'stock_{j}_window_{window}_offset_{offset}'] = df_train[f'stock_{j}'].diff(window).shift(target_forecast + window * offset)
                    df_temp.dropna(inplace = True)
    
                    X = sm.add_constant(df_temp[f'stock_{j}_window_{window}_offset_{offset}'])
                    y = df_temp['target']
                    
                    model = sm.OLS(y, X)
                    results = model.fit(vcov = 'HC3')
    
                    if results.pvalues[f'stock_{j}_window_{window}_offset_{offset}'] < alpha:# and results.rsquared > rsquared_limit:
                        current_stock_signals.append((f'stock_{j}_window_{window}_offset_{offset}',\
                                                      results.params[f'stock_{j}_window_{window}_offset_{offset}'],
                                                     results.params['const']))
        models[f'stock_{i}'] = current_stock_signals
    pass

def predict(df, stock_id):
    score = 0
    for model in models[f'stock_{stock_id}']:
        feature, m, c = model
        forecast = getFeature(df, len(df), feature) * m + c #Calculate Linear Model Estimate
        if forecast > 0:
            score += 1
        elif forecast < 0:
            score -= 1
        
    return score / len(models[f'stock_{stock_id}'])


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    window_size = 1
    lookback_periods = 3
    
    if (nt < 2):
        return np.zeros(nins)

    #Set up dataframe from prices
    df_prices = pd.DataFrame()
    for i in range(nins):
        df_prices[f'stock_{i}'] = prcSoFar[i, :]
    if nt > 61 and nt % 50 == 0 and nt != 500: 
        retrain(df_train = df_prices) #Re train with ALL past data.
        print('re training.')
    #Trade based on predictions
    for i in tradedProducts:
        stock_i_signal = predict(df_prices, i)
        #currentPos[i] += stock_i_signal * 10000 / prcSoFar[i, nt - 1]
        
        if stock_i_signal > 0.2:
            currentPos[i] += 1000 / prcSoFar[i, nt - 1]
        elif stock_i_signal < -0.2:
            currentPos[i] -= 1000 / prcSoFar[i, nt - 1]
        
    return currentPos
