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
TRAIN_LENGTH = 250
START_DATE = 750

# Hyperparmeters
SUSPENSION_LOOKBACK = 5
WRONG_SIGN_THRESHOLD = 0.5
SIGN_CHANGE_THRESHOLD = 0.5
SUSPENSION_DAYS = 4
STOP_LOSS_LOOKBACK = 2
STOP_LOSS_SIZE_FACTOR = 0.25

BUILD_PERIOD = 10

features = ['close_to_close (t-5)(t0)']
TARGET_COLUMN = ''
FIVE_DAY_TARGET_COLUMN = 'forward_5_'


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
            
    return log_return_df

def feature_engineer(prices):
    
    # add market which is the mean of all returns
    prices = np.vstack((prices, prices.mean(axis=0)))

    log_return_df = get_log_returns(prices)

    log_return_df.rename(columns={stock_id:str(stock_id) for stock_id in range(nInst+1)}, inplace=True)
        
    
    return log_return_df

def build_multivariate_linear_regression(log_return_df, target_stock, feature_stock, features, test_start_date, train_length, target_column):

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

def inference_multivariate_linear_regression(log_return_df, model, target_stock, feature_stock, features):
    
    X = log_return_df[[f'{_}_{target_stock}' for _ in features]+[f'{_}_{feature_stock}' for _ in features]].iloc[-1:]
    if target_stock == feature_stock:
        X = log_return_df[[f'{_}_{target_stock}' for _ in features]].iloc[-1:]

    X = X.assign(const=1)
    
    pred_t1 = model.predict(X).values[0]

    return pred_t1

def predict_train_multivariate(log_return_df, model, target_stock, feature_stock, features, test_start_date, train_length, target_column):
    
    data = log_return_df[[f'{target_column}{target_stock}']+[f'{_}_{target_stock}' for _ in features]+[f'{_}_{feature_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    if target_stock == feature_stock:
        data = log_return_df[[f'{target_column}{target_stock}']+[f'{_}_{target_stock}' for _ in features]].iloc[test_start_date-train_length:test_start_date]
    data.dropna(inplace=True)

    y = data[f'{target_column}{target_stock}']
    X = data.drop(columns=[f'{target_column}{target_stock}'])
    X = X.assign(const=1)

    y_pred = model.predict(X)
    
    return y, y_pred

def build_models_for_this_period(log_return_df, test_start_date, features, train_length, target_column):

    good_model_dict = defaultdict(dict)
    model_accuracy_dict = defaultdict(dict)
    model_features_dict = defaultdict(dict)

    for i in range(nInst):
        for j in range(nInst+1):
            
            ma_model = build_multivariate_linear_regression(log_return_df, i, j, features, test_start_date, train_length, target_column)
            ma_y, ma_pred = predict_train_multivariate(log_return_df, ma_model, i, j, features, test_start_date, train_length, target_column)

            if abs(ma_model.tvalues.values[0]) >= 2 or abs(ma_model.tvalues.values[len(features)]//2) >= 2:
                good_model_dict[i][j] = ma_model
                model_features_dict[i][j] = features
                model_accuracy_dict[i][j] = accuracy_score(np.sign(ma_y), np.sign(ma_pred))

    final_model_dict = defaultdict(dict)

    for stock_i in model_accuracy_dict:
        top_10 = sorted(model_accuracy_dict[stock_i].items(), key=lambda x: x[1], reverse=True)
        # need to keep top_10 even
        for stock_j, score in top_10:
            
            if stock_j in good_model_dict[stock_i]: # Todo: in future take average, or take average of votes?
                final_model_dict[stock_i][stock_j] = good_model_dict[stock_i][stock_j]  

    return final_model_dict, model_features_dict

def exp_growth(x, a=500, b= 0.00013863, c=0):
    return a * np.exp(b * x) + c

def get_suspension_signals(log_return_df, prcSoFar, stock_i, START_DATE, SUSPENSION_LOOKBACK, signals):
    lookback_returns = log_return_df[str(stock_i)].iloc[-SUSPENSION_LOOKBACK-1:-1].tolist() # get last 10 days of return
    # get 10 day sign flip
    last_n_return_sign_changes = sum(np.sign(lookback_returns[i]) != np.sign(lookback_returns[i-1]) for i in range(1, len(lookback_returns)))

    # get accuracy
    # get number of signal different sign to last 10 days
    lookback_signals = signals[stock_i, prcSoFar.shape[1]-START_DATE-SUSPENSION_LOOKBACK:prcSoFar.shape[1]-START_DATE]
    wrong_sign = sum(lookback_signals[i] != 0 and np.sign(lookback_signals[i]) != np.sign(lookback_returns[i]) for i in range(len(lookback_signals)))

    return last_n_return_sign_changes, wrong_sign

def apply_suspension(last_n_return_sign_changes, wrong_sign, SUSPENSION_LOOKBACK, SUSPENSION_DAYS, SIGN_CHANGE_THRESHOLD, WRONG_SIGN_THRESHOLD, suspension_days, suspension_activated, stock_i, START_DATE, prcSoFar):
    
    if last_n_return_sign_changes/SUSPENSION_LOOKBACK >= SIGN_CHANGE_THRESHOLD or wrong_sign/SUSPENSION_LOOKBACK >= WRONG_SIGN_THRESHOLD:
        suspension_days[stock_i] = SUSPENSION_DAYS # ban for 3 days
        suspension_activated[stock_i, prcSoFar.shape[1]-START_DATE] = 1
        
    return suspension_days, suspension_activated

def get_stop_loss_signals(currentPos, log_return_df, stock_i, STOP_LOSS_LOOKBACK, final_model_dict_5, model_features_dict_5):
    # get sum of sign of last 2 days
    stop_loss_lookback_return = np.sum(np.sign(log_return_df[str(stock_i)].iloc[-STOP_LOSS_LOOKBACK-1:-1].tolist())) 

    # get current position sign
    stock_i_curr_pos_sign = np.sign(currentPos[stock_i])

    # get 5 days prediction
    stock_i_predictions_list_5 = []
    for stock_j in final_model_dict_5[stock_i]:
        stock_i_predictions_list_5.append(inference_multivariate_linear_regression(log_return_df,
                                final_model_dict_5[stock_i][stock_j],
                                stock_i,
                                stock_j,
                                model_features_dict_5[stock_i][stock_j]))
    
    stock_i_prediction_5 = np.sign(np.sum([np.sign(x) for x in stock_i_predictions_list_5]))

    return stop_loss_lookback_return, stock_i_curr_pos_sign, stock_i_prediction_5

def apply_stop_loss(stop_loss_lookback_return, stock_i_prediction_5, stock_i_curr_pos_sign, STOP_LOSS_LOOKBACK, STOP_LOSS_SIZE_FACTOR, currentPos, prcSoFar, START_DATE, stop_loss_activated, stock_i):
    stop_loss_today = False

    # if stop_loss_lookback_return == -STOP_LOSS_LOOKBACK and stock_i_prediction_5 == -1 and stock_i_curr_pos_sign == 1:
    if stock_i_prediction_5 == -1 and stock_i_curr_pos_sign == 1:
        currentPos[stock_i] -= STOP_LOSS_SIZE_FACTOR * exp_growth(abs(currentPos[stock_i]*prcSoFar[stock_i, -1]))/prcSoFar[stock_i, -1]
        stop_loss_today = True
        stop_loss_activated[stock_i, prcSoFar.shape[1]-START_DATE] = 1
    # elif stop_loss_lookback_return == STOP_LOSS_LOOKBACK and stock_i_prediction_5 == 1 and stock_i_curr_pos_sign == -1:
    elif stock_i_prediction_5 == 1 and stock_i_curr_pos_sign == -1:
        currentPos[stock_i] += STOP_LOSS_SIZE_FACTOR * exp_growth(abs(currentPos[stock_i]*prcSoFar[stock_i, -1]))/prcSoFar[stock_i, -1]
        stop_loss_today = True
        stop_loss_activated[stock_i, prcSoFar.shape[1]-START_DATE] = 1
    
    return currentPos, stop_loss_today, stop_loss_activated

def get_signals(log_return_df, final_model_dict, stock_i, model_features_dict, signals, prcSoFar, START_DATE):
    
    stock_i_predictions_list = []
    for stock_j in final_model_dict[stock_i]:
        stock_i_predictions_list.append(inference_multivariate_linear_regression(log_return_df,
                                final_model_dict[stock_i][stock_j],
                                stock_i,
                                stock_j,
                                model_features_dict[stock_i][stock_j]))

        
    stock_i_prediction = np.sign(np.sum([np.sign(x) for x in stock_i_predictions_list]))

    signals[stock_i, prcSoFar.shape[1]-START_DATE] = stock_i_prediction

    return stock_i_prediction, signals

def update_position_for_stock(stock_i_prediction, currentPos, stock_i, prcSoFar):
    if stock_i_prediction > 0:
        if currentPos[stock_i] > 0:
            currentPos[stock_i] += exp_growth(10000-abs(currentPos[stock_i]*prcSoFar[stock_i, -1]))/prcSoFar[stock_i, -1]
        else:
            currentPos[stock_i] += exp_growth(abs(currentPos[stock_i]*prcSoFar[stock_i, -1]))/prcSoFar[stock_i, -1]
    elif stock_i_prediction < -0:
        if currentPos[stock_i] < 0:
            currentPos[stock_i] -= exp_growth(10000-abs(currentPos[stock_i]*prcSoFar[stock_i, -1]))/prcSoFar[stock_i, -1]
        else:
            currentPos[stock_i] -= exp_growth(abs(currentPos[stock_i]*prcSoFar[stock_i, -1]))/prcSoFar[stock_i, -1]
    
    return currentPos

def get_CI(log_return_df, test_start_date, train_length, stock_i):

    stock_data = log_return_df[f'{stock_i}']

    stock_data.dropna(inplace=True)

    stock_n = len(stock_data)
    stock_mean = np.nanmean(stock_data)
    stock_std = np.nanstd(stock_data)  

    lower_bound, upper_bound = stock_mean + np.array([-1, 1]) * 2 * stock_std / np.sqrt(stock_n)

    if np.sign(lower_bound) == np.sign(upper_bound):
        return np.sign(lower_bound), True
    else:
        return np.nan, False

def get_recent_CI(log_return_df, test_start_date, train_length, stock_i):

    stock_data = log_return_df[f'{stock_i}'].loc[test_start_date-train_length:test_start_date]

    stock_data.dropna(inplace=True)

    stock_n = len(stock_data)
    stock_mean = np.nanmean(stock_data)
    stock_std = np.nanstd(stock_data)  

    lower_bound, upper_bound = stock_mean + np.array([-1, 1]) * 2 * stock_std / np.sqrt(stock_n)

    if np.sign(lower_bound) == np.sign(upper_bound):
        return np.sign(lower_bound), True
    else:
        return np.nan, False

def get_CI_for_this_period(log_return_df, first_start_date, train_length):

    bull_bear_stocks = dict()

    for stock_i in range(nInst):
        sign, bull_bear_signal = get_CI(log_return_df, first_start_date, train_length, stock_i)
        recent_sign, recent_bull_bear_signal = get_recent_CI(log_return_df, first_start_date, train_length, stock_i)
        
        if bull_bear_signal and recent_bull_bear_signal and sign == recent_sign:
            bull_bear_stocks[stock_i] = sign
    
    return bull_bear_stocks

def getMyPosition(prcSoFar):
    
    global currentPos, \
            final_model_dict, \
            model_features_dict, \
            test_start_date, \
            signals, \
            suspension_days, \
            stop_loss_activated, \
            suspension_activated, \
            final_model_dict_5, \
            model_features_dict_5, \
            bull_bear_stocks
    
    # last day don't make any new positions
    if prcSoFar.shape[1] == START_DATE+250: 
        return currentPos

    # feature engineered data
    log_return_df = feature_engineer(prcSoFar)

    # first day train the models
    if prcSoFar.shape[1] == START_DATE:

        # get structures for reporting
        signals = np.zeros([50, 251])
        suspension_activated = np.zeros([50, 251])
        stop_loss_activated = np.zeros([50, 251])

        # structure to record days of suspension for each model 
        suspension_days = defaultdict(int)

        # get the first date for test period
        test_start_date = prcSoFar.shape[1]

        # train models
        final_model_dict, model_features_dict = build_models_for_this_period(log_return_df, test_start_date, features, TRAIN_LENGTH, TARGET_COLUMN)
        final_model_dict_5, model_features_dict_5 = build_models_for_this_period(log_return_df, test_start_date, features, TRAIN_LENGTH, FIVE_DAY_TARGET_COLUMN)

        bull_bear_stocks = get_CI_for_this_period(log_return_df, START_DATE, TRAIN_LENGTH)

    for stock_i in bull_bear_stocks:

        if bull_bear_stocks[stock_i] < 0:
            currentPos[stock_i] = -10000/prcSoFar[stock_i, -1]

    for stock_i in final_model_dict:

        if stock_i in bull_bear_stocks and bull_bear_stocks[stock_i] < 0:
            continue

        stop_loss_today = False

        if prcSoFar.shape[1] >= START_DATE + BUILD_PERIOD: # skip first couple of days
            
            # MODEL SUSPENSION
            last_n_return_sign_changes, wrong_sign = get_suspension_signals(log_return_df, prcSoFar, stock_i, START_DATE, SUSPENSION_LOOKBACK, signals)
            suspension_days, suspension_activated = apply_suspension(last_n_return_sign_changes, wrong_sign, SUSPENSION_LOOKBACK, SUSPENSION_DAYS, SIGN_CHANGE_THRESHOLD, WRONG_SIGN_THRESHOLD, suspension_days, suspension_activated, stock_i, START_DATE, prcSoFar)

            # STOP LOSS
            stop_loss_lookback_return, stock_i_curr_pos_sign, stock_i_prediction_5 = get_stop_loss_signals(currentPos, log_return_df, stock_i, STOP_LOSS_LOOKBACK, final_model_dict_5, model_features_dict_5)
            currentPos, stop_loss_today, stop_loss_activated = apply_stop_loss(stop_loss_lookback_return, stock_i_prediction_5, stock_i_curr_pos_sign, STOP_LOSS_LOOKBACK, STOP_LOSS_SIZE_FACTOR, currentPos, prcSoFar, START_DATE, stop_loss_activated, stock_i)

        # OPEN POSITION WITH MODEL
        stock_i_prediction, signals = get_signals(log_return_df, final_model_dict, stock_i, model_features_dict, signals, prcSoFar, START_DATE)

        # consider suspension and stop loss, then adjust position
        if suspension_days[stock_i] > 0: # model suspended
            suspension_days[stock_i] -= 1
        elif stop_loss_today: # stop loss, don't use model to open position
            pass 
        else: # update the position
            currentPos = update_position_for_stock(stock_i_prediction, currentPos, stock_i, prcSoFar)

        # clip to 10k
        currentPos[stock_i] = np.clip(currentPos[stock_i], -10000/prcSoFar[stock_i, -1], 10000/prcSoFar[stock_i, -1])

    return currentPos