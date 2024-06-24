
import pandas as pd
import numpy as np
from collections import defaultdict 
from statsmodels.regression.linear_model import OLS

import warnings
warnings.filterwarnings("ignore")

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

### HYPERPARAMETERS
TRAIN_LENGTH = 250
TEST_LENGTH = 250


features = ['lag1', 'lag2', 'lag3', 'ma10']
TREND = False

old_model_dict = None

def getMyPosition(prcSoFar):
    global currentPos, good_model_dict, old_model_dict
    log_return_df = feature_engineer(prcSoFar)

    # retrain model every 50 days
    if prcSoFar.shape[1] % TEST_LENGTH == 0:
        first_start_date = prcSoFar.shape[1]

        good_model_dict = build_models_for_this_period(log_return_df, first_start_date)

    #     if prcSoFar.shape[1] != 250: # HARDCODED
    #         for stock_i in old_model_dict:
    #             if stock_i not in good_model_dict:
    #                 currentPos[stock_i] = 0
    
    # old_model_dict = good_model_dict

    # make predictions
    for stock_i in good_model_dict:
        stock_i_predictions_list = []
        for stock_j in good_model_dict[stock_i]:
            stock_i_predictions_list.append(inference_model(log_return_df,
                                       good_model_dict[stock_i][stock_j],
                                       stock_i,
                                       stock_j,
                                       features,
                                       trend = TREND))
        stock_i_prediction = np.mean(stock_i_predictions_list)

        if stock_i_prediction >= 0.0000:
            currentPos[stock_i] += 1000/prcSoFar[stock_i, -1]
        elif stock_i_prediction < -0.0000:
            currentPos[stock_i] -= 1000/prcSoFar[stock_i, -1]

    return currentPos

def build_models_for_this_period(log_return_df, first_start_date):

    good_model_dict = defaultdict(dict)

    for i in range(nInst):
        for j in range(nInst):
            model = model_building(log_return_df, i, j, features, first_start_date, TRAIN_LENGTH, TREND)
             
            if abs(model.tvalues.values[0]) >= 2:
                good_model_dict[i][j] = model

    return good_model_dict

def inference_model(log_return_df, model, target_stock, feature_stock, features, trend = False):
    
    X = log_return_df[[f'{_}_{feature_stock}' for _ in features]].iloc[-1:]

    X = X.assign(const=1)

    if trend:
        X = X.assign(trend=np.arange(len(X))+TRAIN_LENGTH)
    
    pred_t1 = model.predict(X).values[0]

    return pred_t1


def get_log_returns(prices):
    # get log_returns
    # put into pandas
    prices_df = pd.DataFrame(prices).T
    # turn into log returns
    log_return_df = prices_df.pct_change().apply(lambda x: np.log(1+x))
    return log_return_df


def feature_engineer(prices):
    log_return_df = get_log_returns(prices)
    # feature engineering

    for stock_id in range(nInst):
        # create lag1
        log_return_df['lag1_'+str(stock_id)] = log_return_df[stock_id].shift(1)
        # create lag2
        log_return_df['lag2_'+str(stock_id)] = log_return_df[stock_id].shift(2)
        # create lag3
        log_return_df['lag3_'+str(stock_id)] = log_return_df[stock_id].shift(3)
        # create lag4
        log_return_df['lag4_'+str(stock_id)] = log_return_df[stock_id].shift(4)

        # create MA
        log_return_df['ma5_'+str(stock_id)] = log_return_df[stock_id].rolling(window=5).mean().shift(1)
        log_return_df['ma10_'+str(stock_id)] = log_return_df[stock_id].rolling(window=10).mean().shift(1)
        log_return_df['ma20_'+str(stock_id)] = log_return_df[stock_id].rolling(window=20).mean().shift(1)
    
    return log_return_df


def model_building(log_return_df, target_stock, feature_stock, features, test_start_date, train_length, trend = False):

    data = log_return_df[[target_stock]+[f'{_}_{feature_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    data.dropna(inplace=True)

    y = data[target_stock]
    X = data.drop(target_stock, axis=1)
    X = X.assign(const=1)

    if trend:
        X = X.assign(trend=np.arange(len(X))+TRAIN_LENGTH)
        
    # build models
    model = OLS(y, X).fit()

    return model