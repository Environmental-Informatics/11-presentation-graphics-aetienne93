#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:16:37 2020

@author: aetienne
"""

import pandas as pd
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    global DataDF
    global MissingValues 
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # check for neagtive discharge
    DataDF["Discharge"] = DataDF["Discharge"].where( DataDF["Discharge"] >= 0 )
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

 #I learned this in the last assignment, replaces any values that fall below zero
    #with NAN- and it actually works when graphing
def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    DataDF = DataDF.loc[startDate:endDate]

    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )
    
def CalcMonthlyFlowRate(DataDF):
    '''This definition utilizes monthly average (mean) flow from each site number
    in the Wildcat Creek and Tippeconoe River data. Monthly statistics and descriptive 
    measurements are then calculated for the given sites over this time.'''
    colNames = ['site_no', 'Mean Flow']
    sfMonth = DataDF.resample('MS').mean()
    #Set dataframes similar to program_10 
    MoDataDF = pd.DataFrame(0, index=sfMonth.index, columns=colNames)
    MoDataDF['site_no'] = DataDF.resample('MS')['site_no'].mean()
    MoDataDF['Mean Flow'] = DataDF.resample('MS')['Discharge'].mean()
    return(MoDataDF)
    
def CalcAvgMonthlyFlowRate(MoDataDF):
    '''This definition utilizes the previously defined data frame 'MoDataDF' to 
    calculate average monthly streamflow for each site number and loaction. Output 
    is mean values for streamflow given the monthly data dataframe'''

    colNames = ['site_no','Mean Flow']
    MonthlyAverages = pd.DataFrame(0, index=[1,2,3,4,5,6,7,8,9,10,11,12],columns=colNames)
    for i in range(0,12):
        MonthlyAverages.iloc[i,0]=MoDataDF['site_no'][::12].mean() 
        MonthlyAverages.iloc[i,1]=MoDataDF['Mean Flow'][i::12].mean() 

    
    return( MonthlyAverages )
    
def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    #read in file and define DataDF
    DataDF = pd.read_csv(fileName, parse_dates=[0])
    DataDF = DataDF.set_index('Date')
    
    return( DataDF )
    
    # the following condition checks whether we are running as a script, in which 
    # case run the test code, otherwise functions are being imported so do not.
    # put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    
    # create dictionary definitions 
    DataDF = {}
    MoDataDF = {}
    MonthlyAverages = {}
    PeakFlow = {}
    MissingValues = {}

    
    #import the required data from files 
    #Read data function from program_10
    ReadData("TippecanoeRiver_Discharge_03331500_19431001-20200315.txt")
    #Clip data to timeline using function from program_10 
    orgTippe, MissingValues = ClipData(DataDF, '1969-10-01', '2019-09-30')
    #similar to monthly metrics function from program_10, creates Fig 3 data
    monthlysfTippe = CalcMonthlyFlowRate(orgTippe)
    monthlysfTippe = CalcAvgMonthlyFlowRate(monthlysfTippe)
    
    #same thing for wildcat creek 
    ReadData("WildcatCreek_Discharge_03335000_19540601-20200315.txt")
    orgWildcat, MissingValues = ClipData(DataDF, '1969-10-01', '2019-09-30')
    monthlysfWildcat = CalcMonthlyFlowRate(orgWildcat)
    monthlysfWildcat = CalcAvgMonthlyFlowRate(monthlysfWildcat)
    
    #Read in annual and monthly metrics from the metrics defined function 
    AnMet = ReadMetrics('Annual_Metrics.csv')
    MoMet = ReadMetrics('Monthly_Metrics.csv')
    
    #Create a figure of the daily streamflow over the last 5 years
    Tippe5Y = orgTippe['2014-10-01': '2019-09-30']
    Wildcat5Y = orgWildcat['2014-10-01': '2019-09-30']
    plt.figure(figsize=(16,10))
    plt.subplot(211)
    plt.plot(Tippe5Y.index,Tippe5Y['Discharge'], 'black',label = 'Tippecanoe')
    plt.ylabel('Discharge (cfs)')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(Wildcat5Y.index,Wildcat5Y['Discharge'], 'green',label = 'Wildcat')
    plt.xlabel('Date')
    plt.ylabel('Discharge (cfs)')
    plt.legend(loc='upper right') #adding legend
    plt.savefig('Five_year_streamflow.png')
    plt.close()
    
    #Annual statistics for each site (coeff var, TQmean, and R-B index)
    fig = plt.figure(figsize=(16,10)) 
    plt.subplot(311)
    #plot from csv metrics 
    #filter for Tippe
    plt.plot(AnMet.index[AnMet['Station'] == 'Tippe'],
             AnMet['Coeff Var'][AnMet['Station'] == 'Tippe'], 'black', linestyle='None',marker='.', label='Tippecanoe') 
    #filter for Wildcat 
    plt.plot(AnMet.index[AnMet['Station'] == 'Wildcat'],
             AnMet['Coeff Var'][AnMet['Station'] == 'Wildcat'], 'red', linestyle='None',marker='x',label='Wildcat')
    plt.legend(loc='upper right')
    #give the axis an object point
    ax = plt.gca() 
    #remove tickmarks
    ax.axes.xaxis.set_ticklabels([])
    #add vertical lines 
    ax.xaxis.grid(which='major',color='gray',linewidth=0.5,linestyle='--',alpha=0.5) 
    plt.ylabel('Coeffecient Variable')
     #add subplot label
    plt.text(-1,200,'A)')
    plt.subplot(312)
    plt.plot(AnMet.index[AnMet['Station'] == 'Tippe'],
             AnMet['Tqmean'][AnMet['Station'] == 'Tippe'], 'black', linestyle='None',marker='.')
    plt.plot(AnMet.index[AnMet['Station'] == 'Wildcat'],
             AnMet['Tqmean'][AnMet['Station'] == 'Wildcat'], 'red', linestyle='None',marker='x')
    ax = plt.gca() 
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.grid(which='major',color='gray',linewidth=0.5,linestyle='--',alpha=0.5)
    plt.ylabel('Tqmean')
    plt.text(-1,0.5,'B)')
    plt.subplot(313)
    plt.plot(AnMet.index[AnMet['Station'] == 'Tippe'],
             AnMet['R-B Index'][AnMet['Station'] == 'Tippe'], 'black', linestyle='None',marker='.')
    plt.plot(AnMet.index[AnMet['Station'] == 'Wildcat'],
             AnMet['R-B Index'][AnMet['Station'] == 'Wildcat'], 'red', linestyle='None',marker='x')
    #creates x-axis
    ax = plt.gca() 
    #set year values
    ax.set_xticklabels(np.arange(2014,2019,1)) 
    ax.tick_params(axis='x',labelrotation=40) 
    ax.xaxis.grid(which='major',color='gray',linewidth=0.5,linestyle='--',alpha=0.5)
    plt.ylabel('R-B Index')
    plt.text(-1,0.32,'C)')
    plt.savefig('annual_statistics.png')
    plt.close()

    #Average annual monthly streamflow
    fig = plt.figure(figsize=(16,10)) 
    plt.plot(monthlysfTippe.index,monthlysfTippe['Mean Flow'],
             'black',linestyle='None',marker='.',label='Tippecanoe')
    plt.plot(monthlysfWildcat.index,monthlysfWildcat['Mean Flow'],'red',
             linestyle='None',marker='x',label='Wildcat')
    plt.xticks(np.arange(1,13,1)) 
    plt.legend(loc='upper right')
    plt.xlabel('Month of Year')
    plt.ylabel('Discharge (cfs)')
    plt.savefig('average_monthly_streamflow.png')
    plt.close()
    
    #period of annual peak flow events
    peakWildcat = AnMet[AnMet['Station']=='Wildcat']
    peakTippe = AnMet[AnMet['Station'] == 'Tippe']
    #drop unneeded columns or else the plot gets messy 
    exceedence_tippe= peakTippe.drop(columns=['site_no', 'Mean Flow', 'Median Flow', 'Coeff Var', 'Skew', 'Tqmean', 'R-B Index', '7Q', '3xMedian'])
    #sort values for Tippe in descending order 
    tippeFlow = exceedence_tippe.sort_values('Peak Flow', ascending = False)
    #create structure to rank values
    rank_tippe= stats.rankdata(tippeFlow['Peak Flow'], method='average') 
    rank_tippe_2=rank_tippe[::-1]
    exceedence_tippe=[(rank_tippe_2[i]/(len(tippeFlow)+1)) for i in range(len(tippeFlow))]
    '''
    ep_tippe=tippe_met.drop(columns=['site_no', 'Mean Flow', 'Median Flow', 'Coeff Var', 'Skew', 'Tqmean', 'R-B Index', '7Q', '3xMedian'])
    tippe_flow=ep_tippe.sort_values('Peak Flow', ascending=False)
    tippe_ranks1=stats.rankdata(tippe_flow['Peak Flow'], method='average')
    tippe_ranks2=tippe_ranks1[::-1]
    tippe_ep=[(tippe_ranks2[i]/(len(tippe_flow)+1)) for i in range(len(tippe_flow))]
    '''
    
    #assign rank to each value 
    #peakTippe_sort['Rank'] = np.arange(start = 1, stop = 51, step =1) 
    #calc exceedence 
    #peakTippe_sort['Exceedence'] = peakTippe_sort['Rank']/51
    #do the same for wildcat 
    exceedence_wildcat= peakWildcat.drop(columns=['site_no', 'Mean Flow', 'Median Flow', 'Coeff Var', 'Skew', 'Tqmean', 'R-B Index', '7Q', '3xMedian'])
    wildcatFlow = peakWildcat.sort_values('Peak Flow', ascending = False)
    #peakWildcat_sort['Rank'] = np.arange(start = 1, stop = 51, step =1) 
    #peakWildcat_sort['Exceedence'] = peakWildcat_sort['Rank']/51
    #fig = plt.figure(figsize=(12,10))
    
    rank_wildcat=stats.rankdata(wildcatFlow['Peak Flow'], method='average') 
    rank_wildcat_2=rank_wildcat[::-1]
    exceedence_wildcat=[(rank_wildcat[i]/(len(wildcatFlow)+1)) for i in range(len(wildcatFlow))]
    
    '''
    #add exceednece values 
    plt.plot(peakTippe_sort['Exceedence'],
             peakTippe_sort['Peak Flow'], 'black', linestyle='None',marker='.', label = 'Tippecanoe') 
    plt.plot(peakWildcat_sort['Exceedence'],
             peakWildcat_sort['Peak Flow'], 'red', linestyle='None',marker='x', label = 'Wildcat')
    ax = plt.gca()
    #set reversal of the x axis 
    ax.set_xlim(1,0) 
    plt.xlabel('Exceedence Probability')
    plt.ylabel('Discharge (cfs)')
    plt.legend(loc='lower right')
    ax.yaxis.grid(which='major',color='gray',linewidth=0.5,linestyle='--',alpha=0.5) 
    plt.savefig('exceedence_prob.png')
    plt.close()
   '''
   
# Excendence Probability Plot 
    fig = plt.figure(figsize=(16,10)) 
    plt.plot(exceedence_tippe, peakTippe['Peak Flow'], label='Tippecanoe River', color='black')
    plt.plot(exceedence_wildcat, peakWildcat['Peak Flow'], label='Wildcat River', color='green')
    plt.xlabel("Exceedence Probability",fontsize=20)
    plt.ylabel("Peak Discharge (CFS)",fontsize=20)
    ax= plt.gca()
    ax.set_xlim(1,0) #reverse x axis 
    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('ExceedenceProbability.png', dpi=96) #Save plot as PNG with 96 dpi   
    plt.close()
    
    
    
    