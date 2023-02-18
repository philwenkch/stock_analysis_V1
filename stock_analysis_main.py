#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:56:13 2023

@author: philipp
"""

import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import math
from scipy import stats

import plotly.express as px

import datetime as dt
from datetime import datetime, timedelta, date, time
import time
from time import sleep

import os.path
import json

import yfinance as yf
from yahooquery import Ticker as yqT


########################################################################
def getTickerListsFromFolder(a_listDir):
    ## returns saved tickerlist in ./data/tickerlists, List of Lists
    ## it is assumed that ticker lists are stored 
    ##      - in files with ending "*.tlist"
    ##      - are stored in json file format (-> using "json.load" command)
    ## returns a dict with key = filename and values = list of tickers (as string) 
    
    # def a_token
    a_token = '.tlist'
    
    # read a directory listing and remove all files which do not match to a_token
    a_fileList = os.listdir(a_listDir)
    for a_item in a_fileList:
        if not a_token in a_item:
            a_fileList.remove(a_item)

    # read from all files in a_file_list and put Input in dictionary
    a_DictionaryOfTickerlists = {}
    for i_name in a_fileList:
        with open(a_listDir + i_name, "r") as fp:
            a_list = json.load(fp)
        a_DictionaryOfTickerlists[i_name.removesuffix(a_token)] = a_list.copy()
        
        
    return a_DictionaryOfTickerlists


########################################################################
def getTickerInfo(a_ticker, a_InfoList):
    ## retunrs ticker information according to Input, Output as list
    # Input:
    #       a_ticker: ticker symbol as string, i.e. 'ZURN.SW' for Zurich Assurance
    #       a_InfoList: list of attributes to collect (using yfinance package)
    #
    print('Try to get Tickerinfo for ' + a_ticker)
    a_TickerData = yf.Ticker(a_ticker)
    a_result = []
    a_result.append( a_ticker )
    a_current = a_TickerData.info
    for a_info in a_InfoList:
        if a_info != 'currentPrice':
            a_result.append( a_current[a_info] )
        elif a_info == 'currentPrice':
            a_result.append( a_TickerData.fast_info['last_price'] )
        else:
            print('Data not found for label: ' + a_info + ' at Ticker: ' + a_ticker)
    
    print('got Tickerinfo for ' + a_ticker)
    return a_result

########################################################################
def getTickersInfo(a_tickerList, a_InfoList):
    ## returns ticker information of a list of tickers according to Input, Output as DataFrame
    # Input:
    #       a_tickerList: tickers as list of strings, i.e. ['ZURN.SW', 'EVE.SW', ..]
    #       a_InfoList: list of attributes to collect (using yfinance package)
    #
    a_result = []
    for a_ticker in a_tickerList:
        a_result.append( getTickerInfo( a_ticker, a_InfoList ) )
        print(a_ticker + ' done.')
        
    #return as dataframe
    a_df = pd.DataFrame(data = a_result, columns = ['ticker'] + a_InfoList)
    a_df.set_index('ticker')
    return a_df

########################################################################
def getYqTickerInfo(a_ticker_list, a_InfoList):
    ## query yahoo data
    # Input: 
    #        a_ticker_list: format as list [ 'Ticker Symbol 1', .. , 'Ticker Symbol n']
    #        a_InfoList: format as collection {'main_key 1' : ['attriute1', 'attribute2', ..], ..
    #                                             'main_key n' : ['attriute1', 'attribute2', ..]  }
    #
    # Output: a_result_df as dataframe with ticker symbol as index and columns of attributes ] } 
    print('loading from yahooquery ..')
    a_data = yqT(a_ticker_list).all_modules 
    print('loading from yahooquery done.')
    a_result = []
    for a_tick in a_ticker_list:
        a_current = a_data[a_tick]
        a_tick_result = []
        for a_tag in a_InfoList:
            if a_tag in a_current.keys():
                # get result for the current key
                a_current_data_by_key = a_current[a_tag]
                #get values for sub_key
                for a_info in a_InfoList[a_tag]:
                    if a_info in a_current_data_by_key.keys():
                        a_tick_result.append( a_current_data_by_key[a_info] )
                    else:
                        print('following key ' + a_tag + '-' + a_info + ' is not available for ' + a_tick + ' .. putting NaN as Value.')
                        a_tick_result.append( np.nan )  
            else:
                print('following key ' + a_tag + ' is not available for ' + a_tick)
                raise Exception('no repair for this - make sure key exist ..')
                   
                    
        print('done for ' + a_tick + ' ..' )
        a_result.append( [a_tick] + a_tick_result )
    
    #flatten the names in order to name columns in dataframe
    a_result_names = []
    for a_key in a_InfoList:
        for a_subkey in a_InfoList[a_key]:
            #print(a_key)
            a_result_names.append( a_subkey )
    a_result_names = list( ['ticker'] + a_result_names )
        
    #bring it to a panda data frame
    a_result_df = pd.DataFrame(data = list(a_result), columns = a_result_names)
    a_result_df.set_index('ticker')
            
    return a_result_df, a_result_names
    


########################################################################
def getNormalizedTickersInfo(a_TickersInfo, a_norm, a_index):
    ## returns normalized TickersInfo per a_norm, i.e. 'currentPrice', Output as DataFrame
    # Input:
    #       a_TickersInfo: DataFrame of unnormalizedTickersInfo
    #       a_norm: column name for which to normlize the other columns
    #       a_index: List of column names which to normalize
    #
    # Output:
    #       same dataframe with added columns of normalized columns, 
    #       normalized columnes are named with '*_norm'
    #
    #
    a_norm_df = a_TickersInfo.copy()
    
    for ind in a_index:
        a_norm_df[ind+'_norm'] = a_norm_df[ind]/a_norm_df[a_norm]
    return a_norm_df


########################################################################
def getTickersTimeSeries(a_tickerListStr, a_start, a_end, a_interval):
   ## returns time series of the given Input, Output as DataFrame
    a_df = yf.download(a_tickerListStr, a_start, a_end, a_interval, ignore_tz = True, 
                       group_by = 'ticker', repair = False, prepost = False)

    a_head = []
    for i in range(len(a_df.columns)):
        a_head.append( a_df.columns[i][0] + '_' + a_df.columns[i][1] )
        #print(a_head)
        
    a_df.columns = a_head
    return a_df




def main():
#    print('Hello everybody!')
    a_dummy=[]

if __name__ == "__main__":

    ########################################################################
    ## Initial Streamlit settings
    st.set_page_config(
        page_title="Stock Info App",
        page_icon=":owl:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
            }
        )    
    st.title(':owl: :red[Information about Stock Tickers] :chart_with_upwards_trend:')
    st.markdown(' :blue[_showing analyst opinions, using yfinance package, pw, Feb23_]')


    ########################################################################
    ## General Settings for Plots
    with st.sidebar:
        st.title('Menu :red[_General Plot Settings_]')
        st.text('')
        col1, col2 = st.columns(2)
        with col1:
            a_plotly_width = int( st.text_input("Set Plot Width :", 1200))
            a_plotly_msize = int( st.text_input("Set Marker Size:", 4 ))
        with col2:
            a_plotly_height = int( st.text_input("Set Plot Height:", int(a_plotly_width/3) ))
            a_plotly_lwidth = float( st.text_input("Set Line Width:", 1 ))
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')

    ########################################################################
    # Some Test Code at the beginning..
    # Here We are getting ZURN financial information
    # We need to pass FB as argument for that
#    getZHData = yf.Ticker("ZURN.SW")
    # whole python dictionary is printed here
#    print(getZHData.info)


    ########################################################################
    ## Setup possible tickerlists to analyse
    ## get my own tickerlist .. , additional NOTE: not available anymore 'BLS.SW'
    a_tickerListDirectory = './data/tickerlists/'
    a_tickerListChoice = getTickerListsFromFolder(a_tickerListDirectory)
    
    with st.sidebar:
        st.title('Menu :red[_Ticker Overview_]')
        st.text('')        
        ########### 
        my_ticker_choice = st.selectbox('Choose Ticker List:', list(a_tickerListChoice.keys()), 
                                      key = list(a_tickerListChoice.keys())[0] )
        my_ticker_list = a_tickerListChoice[my_ticker_choice]
        ########### 
        st.markdown('## Select shown tickers by Options:')
        col1, col2 = st.columns(2)
        with col1:
            numOfAnalists = float( st.text_input(":red[Number of Analyists Opinion] greater or equal:", 1))
        with col2:
            recomMean = float( st.text_input(":red[Recommandation Mean] lower or equal (value: 1-5):", 3))
        

    #my_ticker_list = ['ZURN.SW','EVE.SW','MBTN.SW', 'ASWN.SW', \
    #            'RCK.V', 'TKA.DE', 'F3C.DE', 'VOW3.DE', 'NOVO-B.CO', \
    #            'ALJXR.PA', 'LIN', 'NEL.OL', 'BLDP', 'UUUU', 'SHOP', \
    #            'ONON', 'BE', 'CLNE', 'RUN', 'SQ', 'EQNR', 'FCEL', \
    #            'QS', 'PCELL.ST', 'PLUG', 'SNPS', 'CWR.L' ]

    #a_ticker_name = 'sp500'
    #tickers = si.tickers_sp500()

    #a_ticker_name = 'nasdaq_II'
    #tickers = si.tickers_nasdaq()

    #a_ticker_name = 'other'
    #tickers = si.tickers_other()

    #a_ticker_name = 'six_II'
    #tickers = swiss_tickers = a_six_list['Ticker'].values

    #get today / as time series should retrieved till today
    a_todayStr = datetime.today().strftime("%Y-%m-%d")


    a_info_list = {'financialData': ['numberOfAnalystOpinions', 'recommendationKey' ,'recommendationMean',
                                     'currentPrice','targetMeanPrice','targetLowPrice','targetHighPrice'],
                   'defaultKeyStatistics' : ['forwardEps']}


    ########################################################################
    ## Setup Info attributes for tickers to be informed of
    a_ticker_info = ['numberOfAnalystOpinions', 'recommendationKey', \
                          'recommendationMean','currentPrice','targetMeanPrice',\
                          'targetLowPrice','targetHighPrice','forwardEps']

    ### get a ticker information
    #a_InfoOfZURN = getTickerInfo('ZURN.SW', a_ticker_info)
    #print(a_InfoOfZURN)
    
    #an output filename
    a_outputFilename = './data/' + a_todayStr + '_recommendations_' + my_ticker_choice +'_yF.csv'
    
    data_load_state = st.text('Loading data... ' + a_outputFilename)
    if not os.path.isfile(a_outputFilename):
        # get a whole Info List for tickers
    ########### set out this routine becaus of problems by yfinance package
    ####   a_InfoListOfTickers = getTickersInfo( my_ticker_list, a_ticker_info )
        a_InfoListOfTickers, a_dummy = getYqTickerInfo(my_ticker_list, a_info_list )
        # add another column
        a_InfoListOfTickers['PricePerEPS'] = a_InfoListOfTickers['currentPrice']/a_InfoListOfTickers['forwardEps']
        print(a_InfoListOfTickers)
        # write the output to a file
        a_InfoListOfTickers.to_csv( a_outputFilename )
    else:
        a_InfoListOfTickers = pd.read_csv( a_outputFilename )

    ########################################################################
    ## Setup Info time series for tickers to be informed of
    ## including some indices to compare
    
    #an output filename
    a_outputFilename = './data/' + a_todayStr +'_' + my_ticker_choice + '_data_myTS_FromTickers_yF.csv'
    
    if not os.path.isfile(a_outputFilename): 
        indeces_to_compare = '^NDX ^GDAXI ^SSMI' 
        a_tickerListStr = ''
        maxLength = len(my_ticker_list)
        for i in my_ticker_list[0:maxLength]:
            a_tickerListStr = a_tickerListStr + ' ' + i
            
        a_tickerListStr = indeces_to_compare + a_tickerListStr
        # get time series dataframe
        print('komme hier durch ..')
        a_TS_FromTickers = getTickersTimeSeries(a_tickerListStr, "2022-01-01", a_todayStr, "1d")
        print(a_TS_FromTickers)
        a_TS_FromTickers.to_csv( a_outputFilename )
    else:
        a_TS_FromTickers = pd.read_csv( a_outputFilename )
        a_TS_FromTickers.set_index('Date', inplace=True)
    data_load_state.text('Loading data...done!')

    ## normalize data to plot and compare on same scale
    a_normInfoListOfTickers = getNormalizedTickersInfo(a_InfoListOfTickers, 'currentPrice',
                                                             ['targetMeanPrice', 'targetLowPrice', 
                                                              'targetHighPrice', 'forwardEps'])
    ## adapt targetLowPrice and targetHighPrice to plot in Errorbar Plot
    
    a_normInfoListOfTickers['targetLowPrice_norm'] = a_normInfoListOfTickers['targetMeanPrice_norm'] - a_normInfoListOfTickers['targetLowPrice_norm']
    a_normInfoListOfTickers['targetHighPrice_norm'] = a_normInfoListOfTickers['targetHighPrice_norm'] - a_normInfoListOfTickers['targetMeanPrice_norm']
    # search nan and set it to zero
    a_normInfoListOfTickers = a_normInfoListOfTickers.fillna(0)
    
    #put the filter values in action 
    a_normToDisplay = a_normInfoListOfTickers.loc[(a_normInfoListOfTickers['numberOfAnalystOpinions'] >= numOfAnalists) &
                                                  (a_normInfoListOfTickers['recommendationMean'] <= recomMean)]

    with st.sidebar:
        st.write('### Numer of tickers fullfilling criteria: ', a_normToDisplay.shape[0],
                 ' out of ', a_normInfoListOfTickers.shape[0])
        #st.columns(1,* )
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
    
    ########################################################################
    ## for streamlit presentation
    
    st.title(':telescope: Menu :red[_Ticker Overview_]')
    st.text('')
    
    if st.checkbox('show tickers data'):
        st.dataframe(a_InfoListOfTickers)
    st.markdown('Data is :red[normalized] - :red[current Price] for each ticker is equal :red[1]')

    ########################################################################
    ## doing a plot with matplotlib
    #fig, ax = plt.subplots(figsize=(12,4))
    #x = a_normInfoListOfTickers['ticker']
    #y = a_normInfoListOfTickers['targetMeanPrice']
    #a_error = [a_normInfoListOfTickers['targetLowPrice'], a_normInfoListOfTickers['targetHighPrice']]
    #ax.errorbar(x,y,a_error, marker = 'o', markersize = 6, linestyle='dotted', linewidth=1, capsize=4 )
    #ax.grid(color='lightgray', linestyle=':', linewidth=1)
    #ax.set_title('normalized analyst information per ticker [low, mean, high]')
    #### rotates labels 
    #plt.setp( ax.xaxis.get_majorticklabels(), rotation=90 ) 
    #st.pyplot(fig, None)
    
    ########################################################################
    ## doing a plotly 
    figp = px.scatter(a_normToDisplay, x = 'ticker', y = 'targetMeanPrice_norm', 
                      error_y='targetHighPrice_norm', error_y_minus='targetLowPrice_norm', 
                      color = 'recommendationKey', size = 'numberOfAnalystOpinions',
                      hover_data=a_ticker_info, 
                      labels={'x': 'ticker name', 'y':'normalized Mean'} )
    figp.update_layout(autosize=True)
    figp.update_layout(width = a_plotly_width, height = a_plotly_height)
    figp.add_shape( # add a horizontal "target" line
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
    x0=0, x1=1, xref="paper", y0=1, y1=1, yref="y")
    #figp.update_traces(marker=dict(color='red'))
    figp.update_traces(line_color='red')
    #figp.update_layout(width=800, height=600)
    st.write(figp)
    
    with st.sidebar:
        st.title('Menu :red[_Ticker Time Serie_]')
        st.text('')
        st.markdown('## select ticker for time serie:')
        a_filter_object = filter(lambda a: 'Close' in a, a_TS_FromTickers.columns.values)
        a_selectionList = list(a_filter_object)
        a_selectedTickers = st.multiselect('select tickers', a_selectionList, 
                                                   ['^NDX_Close','^GDAXI_Close','^SSMI_Close'])
        a_startTime, a_endTime = st.select_slider('select range to compare:', list(a_TS_FromTickers.index), 
                                                   value = [list(a_TS_FromTickers.index)[0], 
                                                           list(a_TS_FromTickers.index)[len(list(a_TS_FromTickers.index))-1]] )
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
    
    st.title(':crystal_ball: Menu :red[_Ticker Time Serie_]')
    st.text('')
    if st.checkbox('show ticker time series data'):
        st.dataframe(a_TS_FromTickers[a_startTime : a_endTime])
    
    
    # show the selected tickers chart
    if st.checkbox('show selected ticker time series plot'):
        #st.markdown('Selected Tickers times serie data')
        #st.line_chart( a_TS_FromTickers, y = a_selectedTickers )
        ########################################################################
        ## doing a plotly
        figpt = px.scatter(a_TS_FromTickers[a_startTime : a_endTime], y = a_selectedTickers)
        figpt.update_traces(mode='markers+lines',
                            marker = dict(size = a_plotly_msize),
                            line = dict(width=a_plotly_lwidth, dash='dot')
                            )
        figpt.update_layout(autosize=True)
        figpt.update_layout(width = a_plotly_width, height = a_plotly_height)
        st.write(figpt)
    
    
    

    #normalize a_TS_FromTickers[a_selectedTickers]
    a_normalizedTS_FromTickers = a_TS_FromTickers[a_startTime : a_endTime][a_selectedTickers]/a_TS_FromTickers[a_startTime : a_endTime][a_selectedTickers].iloc[0]
    
    if st.checkbox('show normalized ticker time series data'):
        st.dataframe(a_normalizedTS_FromTickers)
        
    # show the selected tickers chart
    st.markdown('Selected Tickers times serie data :red[normalized]')
    #st.line_chart( a_normalizedTS_FromTickers, y = a_selectedTickers )
    ########################################################################
    ## doing a plotly
    figptnorm = px.scatter(a_normalizedTS_FromTickers, y = a_selectedTickers)
    figptnorm.update_traces(mode='markers+lines', 
                            marker = dict(size = a_plotly_msize),
                            line = dict(width=a_plotly_lwidth, dash='dot')
                            )
    figptnorm.update_layout(autosize=True)
    figptnorm.update_layout(width = a_plotly_width, height = a_plotly_height)
    st.write(figptnorm)
    
    
    
    ########################################################################
    st.title(':mag: Menu :red[_Ticker Correlation_]')
    st.text('')
    
        
    with st.sidebar:
        st.title(':mag: Menu :red[_Ticker Correlation_]')
        col1, col2 = st.columns(2)
        a_filter_object = filter(lambda a: 'Close' in a, a_TS_FromTickers.columns.values)
        a_selectionList = list(a_filter_object)
        with col1:
            a_corrTicker1 = st.selectbox('select ticker 1', a_selectionList)
            a_corrTicker1Data = a_TS_FromTickers[a_startTime : a_endTime][a_corrTicker1]
        with col2:
            a_corrTicker2 = st.selectbox('select ticker 2', a_selectionList)
            a_corrTicker2Data = a_TS_FromTickers[a_startTime : a_endTime][a_corrTicker2]
    if a_corrTicker1 == a_corrTicker2:
        a_corrTicker2 = a_corrTicker1 + '_'
        a_corrTicker2Data.name = a_corrTicker2
    
    a_corrTickerResult = pd.concat([a_corrTicker1Data, a_corrTicker2Data], axis=1).reindex(a_corrTicker1Data.index)
    a_corrTickerResult.dropna(inplace=True, axis=0)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(a_corrTickerResult[a_corrTicker1], 
                                                                  a_corrTickerResult[a_corrTicker2])
    
    
    with st.sidebar:
       st.text( 'slope : {:2.2E} +/- {:2.2E}'.format(slope, std_err) )
       st.text( 'p-value for Hypt. "no slope": {:2.2E}'.format(p_value) )
       st.text( 'r-value: {:4.2f}'.format(r_value) )
    
    if st.checkbox('show correlation line'):
        a_corrTickerResult['corrLine'] = intercept + slope*a_corrTickerResult[a_corrTicker1]
        figcorr = px.scatter(a_corrTickerResult, x = a_corrTicker1, y = [a_corrTicker2, 'corrLine'])
    else:
        figcorr = px.scatter(a_corrTickerResult, x = a_corrTicker1, y = a_corrTicker2)
        
    figcorr.update_traces(mode='markers+lines', 
                            marker = dict(size = a_plotly_msize),
                            line = dict(width=a_plotly_lwidth, dash='dot')
                            )
    figcorr.update_layout(autosize=True)
    figcorr.update_layout(width = a_plotly_width, height = a_plotly_height)
    st.write(figcorr)


















