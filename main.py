import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

from statsmodels.regression.linear_model import OLS
from sklearn.metrics import r2_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

### HYPERPARAMETERS
BACKBONE_MODEL = 'multivariate'

TRAIN_LENGTH = 250
TEST_LENGTH = 250
START_DATE = 250

features = ['close_to_close (t-5)(t0)']
target_column = ''

old_model_dict = None

def getMyPosition(prcSoFar):
    
    global currentPos, final_model_dict, old_model_dict, model_features_dict, first_start_date

    
    if prcSoFar.shape[1] == START_DATE+250: # last day don't make any new positions
        return currentPos

    log_return_df = feature_engineer(prcSoFar)

    # retrain model every 50 days
    if prcSoFar.shape[1] % TEST_LENGTH == 0:
        first_start_date = prcSoFar.shape[1]

        final_model_dict, model_features_dict = build_models_for_this_period(log_return_df, first_start_date)

    #     if prcSoFar.shape[1] != 250: # TODO: hardcoded restore no-trade positions to 0
    #         for stock_i in old_model_dict:
    #             if stock_i not in final_model_dict:
    #                 currentPos[stock_i] = 0
    
        # old_model_dict = final_model_dict
        
    # make predictions
    for stock_i in final_model_dict:
        stock_i_predictions_list = []
        for stock_j in final_model_dict[stock_i]:
            if BACKBONE_MODEL == 'univariate':
                stock_i_predictions_list.append(inference_univariate_linear_regression(log_return_df,
                                       final_model_dict[stock_i][stock_j],
                                       stock_i,
                                       stock_j,
                                       model_features_dict[stock_i][stock_j],
                                       first_start_date,
                                       TRAIN_LENGTH))
            elif BACKBONE_MODEL == 'multivariate':
                stock_i_predictions_list.append(inference_multivariate_linear_regression(log_return_df,
                                        final_model_dict[stock_i][stock_j],
                                        stock_i,
                                        stock_j,
                                        model_features_dict[stock_i][stock_j],
                                        first_start_date,
                                        TRAIN_LENGTH))
        stock_i_prediction = np.sum([np.sign(x) for x in stock_i_predictions_list])


        if stock_i_prediction > 0:
            currentPos[stock_i] += 1000/prcSoFar[stock_i, -1]
        elif stock_i_prediction < -0:
            currentPos[stock_i] -= 1000/prcSoFar[stock_i, -1]
        
    return currentPos

def build_models_for_this_period(log_return_df, first_start_date):

    good_model_dict = defaultdict(dict)
    model_accuracy_dict = defaultdict(dict)
    model_features_dict = defaultdict(dict)

    for i in range(nInst):
        for j in range(nInst+1):
            
            if BACKBONE_MODEL == 'univariate':
                ma_model = build_univariate_linear_regression(log_return_df, i, j, features, first_start_date, TRAIN_LENGTH)
                ma_y, ma_pred = predict_train_univariate(log_return_df, ma_model, i, j, features, first_start_date, TRAIN_LENGTH)
                if abs(ma_model.tvalues.values[0]) >= 2:
                    good_model_dict[i][j] = ma_model
                    model_features_dict[i][j] = features
                    model_accuracy_dict[i][j] = accuracy_score(np.sign(ma_y), np.sign(ma_pred))

            elif BACKBONE_MODEL == 'multivariate':
                ma_model = build_multivariate_linear_regression(log_return_df, i, j, features, first_start_date, TRAIN_LENGTH)
                ma_y, ma_pred = predict_train_multivariate(log_return_df, ma_model, i, j, features, first_start_date, TRAIN_LENGTH)

                if abs(ma_model.tvalues.values[0]) >= 2 or abs(ma_model.tvalues.values[len(features)]//2) >= 2:
                    good_model_dict[i][j] = ma_model
                    model_features_dict[i][j] = features
                    model_accuracy_dict[i][j] = accuracy_score(np.sign(ma_y), np.sign(ma_pred))

    final_model_dict = defaultdict(dict)

    for stock_i in model_accuracy_dict:
        top_10 = sorted(model_accuracy_dict[stock_i].items(), key=lambda x: x[1], reverse=True)
        for stock_j, score in top_10:
            
            if stock_j in good_model_dict[stock_i]: # Todo: in future take average, or take average of votes?
                final_model_dict[stock_i][stock_j] = good_model_dict[stock_i][stock_j]

    return final_model_dict, model_features_dict

def build_multivariate_linear_regression(log_return_df, target_stock, feature_stock, features, test_start_date, train_length):

    data = log_return_df[[f'{target_column}{target_stock}']+[f'{_}_{target_stock}' for _ in features]+[f'{_}_{feature_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    if target_stock == feature_stock:
        data = log_return_df[[f'{target_column}{target_stock}']+[f'{_}_{target_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    data.dropna(inplace=True)

    y = data[f'{target_column}{target_stock}']
    X = data.drop(columns=[f'{target_column}{target_stock}'])
    X = X.assign(const=1)

    # build models
    model = OLS(y, X).fit()

    return model

def inference_multivariate_linear_regression(log_return_df, model, target_stock, feature_stock, features, test_start_date, train_length):
    
    X = log_return_df[[f'{_}_{target_stock}' for _ in features]+[f'{_}_{feature_stock}' for _ in features]].iloc[-1:]
    if target_stock == feature_stock:
        X = log_return_df[[f'{_}_{target_stock}' for _ in features]].iloc[-1:]

    X = X.assign(const=1)
    
    pred_t1 = model.predict(X).values[0]

    return pred_t1

def predict_train_multivariate(log_return_df, model, target_stock, feature_stock, features, test_start_date, train_length):
    
    data = log_return_df[[f'{target_column}{target_stock}']+[f'{_}_{target_stock}' for _ in features]+[f'{_}_{feature_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    if target_stock == feature_stock:
        data = log_return_df[[f'{target_column}{target_stock}']+[f'{_}_{target_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    data.dropna(inplace=True)

    y = data[f'{target_column}{target_stock}']
    X = data.drop(columns=[f'{target_column}{target_stock}'])
    X = X.assign(const=1)

    y_pred = model.predict(X)
    
    return y, y_pred

def get_log_returns(prices):
    # get log_returns
    # put into pandas
    prices_df = pd.DataFrame(prices).T
    # turn into log returns
    log_return_df = prices_df.pct_change().apply(lambda x: np.log(1+x)).shift(-1)

    for ma in [5]:
        for stock_i in range(nInst+1):
            log_return_df[f'forward_{ma}_{stock_i}'] = np.log(prices_df[stock_i]/prices_df[stock_i].shift(ma)).shift(-ma)
            log_return_df[f'close_to_close (t-{ma})(t0)_'+str(stock_i)] = np.log(prices_df[stock_i]/prices_df[stock_i].shift(ma))
            
    # for ma in [1, 5]:
    #     for stock_i in range(nInst):
    #         for stock_j in range(stock_i+1, nInst):
    #             log_return_df[f'diff_forward_{ma}_{stock_i}_{stock_j}'] = log_return_df[f'forward_{ma}_{stock_i}'] - log_return_df[f'forward_{ma}_{stock_j}']
                # log_return_df[f'diff_close_to_close (t-{ma})(t0)_'+str(stock_i)+'_'+str(stock_j)] = log_return_df[f'close_to_close (t-{ma})(t0)_{stock_i}'] - log_return_df[f'close_to_close (t-{ma})(t0)_{stock_j}']

    return log_return_df

def feature_engineer(prices):
    
    # add market which is the mean of all returns
    prices = np.vstack((prices, prices.mean(axis=0)))

    log_return_df = get_log_returns(prices)
    # feature engineering

    for stock_id in range(nInst+1):
        # create lags
        # log_return_df['lag1_'+str(stock_id)] = log_return_df[stock_id].shift(1)
        # log_return_df['lag2_'+str(stock_id)] = log_return_df[stock_id].shift(2)
        
        # create MA
        log_return_df['ma5_'+str(stock_id)] = log_return_df[stock_id].rolling(window=5).mean().shift(1)
        # log_return_df['ma10_'+str(stock_id)] = log_return_df[stock_id].rolling(window=10).mean().shift(1)
        # log_return_df['ma20_'+str(stock_id)] = log_return_df[stock_id].rolling(window=20).mean().shift(1)

    log_return_df.rename(columns={stock_id:str(stock_id) for stock_id in range(nInst+1)}, inplace=True)
        
    
    return log_return_df