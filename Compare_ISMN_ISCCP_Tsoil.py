#!/usr/bin/env /Users/rdcrljbe/anaconda3/bin/python

import os
import numpy as np
import datetime
import pandas
import csv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from scipy import stats
import numpy.ma as ma
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
from matplotlib import gridspec

num_years=6
year_array=[2010, 2011, 2012, 2013, 2014, 2015]

Beg_YYYY = year_array[0]
End_YYYY = year_array[num_years-1]

Beg_mm=1
Beg_dd=1
Beg_HH=0

End_mm=12
End_dd=31
End_HH=23

end_DTG=datetime.datetime(End_YYYY, End_mm, End_dd, End_HH)
EDATE='{:%Y%m%d}'.format(end_DTG)
EDATETXT='{:%Y/%m/%d}'.format(end_DTG)
beg_DTG=datetime.datetime(Beg_YYYY, Beg_mm, Beg_dd, Beg_HH)
BGDATE='{:%Y%m%d}'.format(beg_DTG)
BGDATETXT='{:%Y/%m/%d}'.format(beg_DTG)
date_diff=end_DTG-beg_DTG
num_years = End_YYYY-Beg_YYYY+1
the_hour_csv_range=(date_diff.days+1)*num_years
the_pts_array=np.zeros((((date_diff.days+1)*8)+1)*num_years, dtype=np.int32)

INC_YEAR=Beg_YYYY

ISMNPATH='/Users/rdcrljbe/Data/ISMN/'

#DataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_GRASSLAND/'
#ISCCPDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_GRASSLAND/'
#img_out_path='/Users/rdcrljbe/Data/DIS_test2/output/CONUS_GRASSLAND/'
#stationsfilename=ISMNPATH+'GIS_Data/ISMN_Stations_CONUS_2010_GRASSLAND.csv'
#Plot_Labels='GRASSLAND'
#Plot_Labels_full='GRASSLAND LAND COVER SITES'

#DataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_SHRUBLAND/'
#ISCCPDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_SHRUBLAND/'
#img_out_path='/Users/rdcrljbe/Data/DIS_test2/output/CONUS_SHRUBLAND/'
#stationsfilename=ISMNPATH+'GIS_Data/ISMN_Stations_SWUS_2010_SHRUBLAND.csv'
#Plot_Labels='SHRUBLAND'
#Plot_Labels_full='SHRUBLAND LAND COVER SITES'

#DataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_CROPLAND/'
#ISCCPDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_CROPLAND/'
#img_out_path='/Users/rdcrljbe/Data/DIS_test2/output/CONUS_CROPLAND/'
#stationsfilename=ISMNPATH+'GIS_Data/ISMN_Stations_CONUS_2010_CROPLAND.csv'
#Plot_Labels='CROPLAND'
#Plot_Labels_full='CROPLAND LAND COVER SITES'

DataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_ALL/'
ISCCPDataPath='/Users/rdcrljbe/Data/DIS_test2/STATS_TABLES/CONUS_ALL/'
img_out_path='/Users/rdcrljbe/Data/DIS_test2/output/CONUS_ALL/'
stationsfilename=ISMNPATH+'GIS_Data/ISMN_Stations_All_2013_merged.csv'
Plot_Labels='ALL_SITES'
Plot_Labels_full='ALL SITES'

#check output directory to make sure it exists
if not os.path.exists(img_out_path):
    print ('creating output directory')
    os.mkdir(img_out_path)

print (stationsfilename)
stationsfile=pandas.read_csv(stationsfilename)


for year_loop in range (0, num_years, 1):

    print ('the year loop number is', year_loop, num_years)
    INC_YEAR = year_array[year_loop]
    print ('the year we are processing is', INC_YEAR)
    STR_YR=str(INC_YEAR)
    
    STR_BDATE=STR_YR+'0101'
    STR_EDATE=STR_YR+'1231'
    
    SCAN_LEVELS=[1,2,3,4,5]
    LVL=0
    
    while LVL<=3:
        CURR_SCANfile=DataPath+'ISMN_tsoil_Noah-L'+str(SCAN_LEVELS[LVL])+'_data_by_hour_'+STR_BDATE+'-'+STR_EDATE+'.csv'
        print (DataPath+'ISMN_tsoil_Noah-L'+str(SCAN_LEVELS[LVL])+'_data_by_hour_'+STR_BDATE+'-'+STR_EDATE+'.csv')
        CURR_SCANDATA=pandas.read_csv(CURR_SCANfile)
        num_scan_stations=CURR_SCANDATA.shape[1]
        num_scan_recs=CURR_SCANDATA.shape[0]
        if LVL==0:
            SCAN_DATA_ARRAY_TEMP=np.zeros((4, num_scan_recs, num_scan_stations), dtype=np.float64)
        ii=0
        while ii<=num_scan_stations-2:
            
            SCAN_DATA_ARRAY_TEMP[LVL,:,ii]=CURR_SCANDATA[CURR_SCANDATA.columns[ii+1]]
            ii+=1
        LVL+=1
        
    index_array=np.arange(num_scan_recs-1)
    SCAN_DATA_ARRAY_TEMP=np.nan_to_num(SCAN_DATA_ARRAY_TEMP, copy=True, nan=-9999.0)
    
    if year_loop == 0:
        SCAN_DATA_ARRAY=SCAN_DATA_ARRAY_TEMP
        print ("testing before the concatenation feature", SCAN_DATA_ARRAY.shape, SCAN_DATA_ARRAY_TEMP.shape)
    else:
        SCAN_DATA_ARRAY=np.concatenate((SCAN_DATA_ARRAY, SCAN_DATA_ARRAY_TEMP), axis=1)
        print ("testing the concatenation feature", SCAN_DATA_ARRAY.shape, SCAN_DATA_ARRAY_TEMP.shape)
    
    #####  Read in the Skin Temperature data from the ISCCP files
    
    CURR_ISCCP_TSKIN_file=ISCCPDataPath+'ISCCP_TSKIN_'+STR_BDATE+'-'+STR_EDATE+'.csv'
    CURR_ISCCP_TSKIN_data=pandas.read_csv(CURR_ISCCP_TSKIN_file)
    num_lis_stations=CURR_ISCCP_TSKIN_data.shape[1]
    num_lis_recs=CURR_ISCCP_TSKIN_data.shape[0]
    ISCCP_TSKIN_ARRAY_TEMP=np.zeros((num_lis_recs, num_lis_stations), dtype=np.float64)
    ii=0
    while ii<=num_lis_stations-2:
        ISCCP_TSKIN_ARRAY_TEMP[:,ii]=CURR_ISCCP_TSKIN_data[CURR_ISCCP_TSKIN_data.columns[ii+1]]
        ii+=1
    
    
    if year_loop == 0:
        ISCCP_TSKIN_ARRAY=ISCCP_TSKIN_ARRAY_TEMP
    else:
        ISCCP_TSKIN_ARRAY=np.concatenate((ISCCP_TSKIN_ARRAY, ISCCP_TSKIN_ARRAY_TEMP), axis=0)

##############################################################
## TSKIN Data Arrays for station observation locations
##############################################################

SCAN_three_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_zero_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_six_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_nine_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_twelve_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_fifteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_eighteen_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
SCAN_twtyone_array_soiltemp = np.zeros((4,int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)

##############################################################
## TSKIN Data Arrays for ISCCP locations
##############################################################

ISSCP_three_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_zero_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_six_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_nine_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_twelve_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_fifteen_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_eighteen_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)
ISSCP_twtyone_array_soiltemp = np.zeros((int(the_hour_csv_range/num_years)+1, num_scan_stations), dtype=np.float64)

the_zero_hour_csv_index=0
the_three_hour_csv_index=0
the_six_hour_csv_index=0
the_nine_hour_csv_index=0
the_twlv_hour_csv_index=0
the_fftn_hour_csv_index=0
the_eithn_hour_csv_index=0
the_twtone_hour_csv_index=0

dtg_array_zero=[]
dtg_array_three=[]
dtg_array_six=[]
dtg_array_nine=[]
dtg_array_twelve=[]
dtg_array_fifteen=[]
dtg_array_eighteen=[]
dtg_array_twtyone=[]

Jan_Inc=0
Feb_Inc=0
Mar_Inc=0
Apr_Inc=0
May_Inc=0
Jun_Inc=0
Jul_Inc=0
Aug_Inc=0
Sep_Inc=0
Oct_Inc=0
Nov_Inc=0
Dec_Inc=0

Jan_00_Inc=0
Feb_00_Inc=0
Mar_00_Inc=0
Apr_00_Inc=0
May_00_Inc=0
Jun_00_Inc=0
Jul_00_Inc=0
Aug_00_Inc=0
Sep_00_Inc=0
Oct_00_Inc=0
Nov_00_Inc=0
Dec_00_Inc=0

Jan_03_Inc=0
Feb_03_Inc=0
Mar_03_Inc=0
Apr_03_Inc=0
May_03_Inc=0
Jun_03_Inc=0
Jul_03_Inc=0
Aug_03_Inc=0
Sep_03_Inc=0
Oct_03_Inc=0
Nov_03_Inc=0
Dec_03_Inc=0

Jan_06_Inc=0
Feb_06_Inc=0
Mar_06_Inc=0
Apr_06_Inc=0
May_06_Inc=0
Jun_06_Inc=0
Jul_06_Inc=0
Aug_06_Inc=0
Sep_06_Inc=0
Oct_06_Inc=0
Nov_06_Inc=0
Dec_06_Inc=0

Jan_09_Inc=0
Feb_09_Inc=0
Mar_09_Inc=0
Apr_09_Inc=0
May_09_Inc=0
Jun_09_Inc=0
Jul_09_Inc=0
Aug_09_Inc=0
Sep_09_Inc=0
Oct_09_Inc=0
Nov_09_Inc=0
Dec_09_Inc=0

Jan_12_Inc=0
Feb_12_Inc=0
Mar_12_Inc=0
Apr_12_Inc=0
May_12_Inc=0
Jun_12_Inc=0
Jul_12_Inc=0
Aug_12_Inc=0
Sep_12_Inc=0
Oct_12_Inc=0
Nov_12_Inc=0
Dec_12_Inc=0

Jan_15_Inc=0
Feb_15_Inc=0
Mar_15_Inc=0
Apr_15_Inc=0
May_15_Inc=0
Jun_15_Inc=0
Jul_15_Inc=0
Aug_15_Inc=0
Sep_15_Inc=0
Oct_15_Inc=0
Nov_15_Inc=0
Dec_15_Inc=0

Jan_18_Inc=0
Feb_18_Inc=0
Mar_18_Inc=0
Apr_18_Inc=0
May_18_Inc=0
Jun_18_Inc=0
Jul_18_Inc=0
Aug_18_Inc=0
Sep_18_Inc=0
Oct_18_Inc=0
Nov_18_Inc=0
Dec_18_Inc=0

Jan_21_Inc=0
Feb_21_Inc=0
Mar_21_Inc=0
Apr_21_Inc=0
May_21_Inc=0
Jun_21_Inc=0
Jul_21_Inc=0
Aug_21_Inc=0
Sep_21_Inc=0
Oct_21_Inc=0
Nov_21_Inc=0
Dec_21_Inc=0

Total_Month_days=np.zeros((12), dtype='i4')
Winter_days=np.zeros((3), dtype='i4')
Spring_days=np.zeros((3), dtype='i4')
Summer_days=np.zeros((3), dtype='i4')
Fall_days=np.zeros((3), dtype='i4')

for year_loop in range (0, num_years, 1):
    #check the year to see if it is a leap year
    temp_year=datetime.datetime(year_array[year_loop], 1, 1)
    temp_date_check='{:%Y-%m-%d}'.format(temp_year)
    check_leap_year=pandas.DatetimeIndex([temp_date_check])
    if check_leap_year.is_leap_year == True:
        print (year_array[year_loop], 'a leap year')
        Mon_Days=np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    else:
        print (year_array[year_loop], 'not a leap year')
        Mon_Days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    
    Total_Month_days=Total_Month_days[:]+Mon_Days[:]
    Winter_days=Winter_days+[Mon_Days[11], Mon_Days[0], Mon_Days[1]]
    Spring_days=Spring_days+[Mon_Days[2], Mon_Days[3], Mon_Days[4]]
    Summer_days=Summer_days+[Mon_Days[5], Mon_Days[6], Mon_Days[7]]
    Fall_days=Fall_days+[Mon_Days[8], Mon_Days[9], Mon_Days[10]]
    
num_Winter_days=np.sum(Winter_days)
num_Spring_days=np.sum(Spring_days)
num_Summer_days=np.sum(Summer_days)
num_Fall_days=np.sum(Fall_days)
tot_num_hours=8*np.sum(Total_Month_days)
tot_num_days=np.sum(Total_Month_days)

#  create the SCAN monthly and seasonal arrays

Jan_SCAN=np.zeros((4, Total_Month_days[0]*8, num_scan_stations), dtype=np.float64)
Feb_SCAN=np.zeros((4, Total_Month_days[1]*8, num_scan_stations), dtype=np.float64)
Mar_SCAN=np.zeros((4, Total_Month_days[2]*8, num_scan_stations), dtype=np.float64)
Apr_SCAN=np.zeros((4, Total_Month_days[3]*8, num_scan_stations), dtype=np.float64)
May_SCAN=np.zeros((4, Total_Month_days[4]*8, num_scan_stations), dtype=np.float64)
Jun_SCAN=np.zeros((4, Total_Month_days[5]*8, num_scan_stations), dtype=np.float64)
Jul_SCAN=np.zeros((4, Total_Month_days[6]*8, num_scan_stations), dtype=np.float64)
Aug_SCAN=np.zeros((4, Total_Month_days[7]*8, num_scan_stations), dtype=np.float64)
Sep_SCAN=np.zeros((4, Total_Month_days[8]*8, num_scan_stations), dtype=np.float64)
Oct_SCAN=np.zeros((4, Total_Month_days[9]*8, num_scan_stations), dtype=np.float64)
Nov_SCAN=np.zeros((4, Total_Month_days[10]*8, num_scan_stations), dtype=np.float64)
Dec_SCAN=np.zeros((4, Total_Month_days[11]*8, num_scan_stations), dtype=np.float64)
Winter_SCAN=np.zeros((4, num_Winter_days*8, num_scan_stations), dtype=np.float64)
Spring_SCAN=np.zeros((4, num_Spring_days*8, num_scan_stations), dtype=np.float64)
Summer_SCAN=np.zeros((4, num_Summer_days*8, num_scan_stations), dtype=np.float64)
Fall_SCAN=np.zeros((4, num_Fall_days*8, num_scan_stations), dtype=np.float64)

Jan_00_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_00_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_00_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_00_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_00_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_00_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_00_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_00_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_00_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_00_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_00_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_00_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_00_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_00_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_00_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_00_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_03_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_03_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_03_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_03_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_03_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_03_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_03_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_03_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_03_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_03_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_03_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_03_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_03_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_03_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_03_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_03_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_06_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_06_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_06_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_06_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_06_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_06_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_06_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_06_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_06_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_06_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_06_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_06_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_06_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_06_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_06_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_06_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_09_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_09_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_09_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_09_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_09_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_09_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_09_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_09_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_09_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_09_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_09_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_09_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_09_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_09_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_09_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_09_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_12_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_12_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_12_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_12_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_12_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_12_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_12_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_12_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_12_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_12_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_12_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_12_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_12_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_12_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_12_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_12_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_15_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_15_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_15_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_15_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_15_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_15_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_15_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_15_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_15_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_15_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_15_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_15_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_15_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_15_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_15_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_15_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_18_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_18_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_18_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_18_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_18_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_18_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_18_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_18_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_18_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_18_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_18_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_18_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_18_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_18_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_18_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_18_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_21_SCAN=np.zeros((4, Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_21_SCAN=np.zeros((4, Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_21_SCAN=np.zeros((4, Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_21_SCAN=np.zeros((4, Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_21_SCAN=np.zeros((4, Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_21_SCAN=np.zeros((4, Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_21_SCAN=np.zeros((4, Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_21_SCAN=np.zeros((4, Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_21_SCAN=np.zeros((4, Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_21_SCAN=np.zeros((4, Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_21_SCAN=np.zeros((4, Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_21_SCAN=np.zeros((4, Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_21_SCAN=np.zeros((4, num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_21_SCAN=np.zeros((4, num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_21_SCAN=np.zeros((4, num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_21_SCAN=np.zeros((4, num_Fall_days, num_scan_stations), dtype=np.float64)

#  creae the ISCCP monthly and seasonal arrays

mon_ISCCP_avg=np.zeros((12, num_scan_stations), dtype=np.float64)
Jan_ISCCP=np.zeros((Total_Month_days[0]*8, num_scan_stations), dtype=np.float64)
Feb_ISCCP=np.zeros((Total_Month_days[1]*8, num_scan_stations), dtype=np.float64)
Mar_ISCCP=np.zeros((Total_Month_days[2]*8, num_scan_stations), dtype=np.float64)
Apr_ISCCP=np.zeros((Total_Month_days[3]*8, num_scan_stations), dtype=np.float64)
May_ISCCP=np.zeros((Total_Month_days[4]*8, num_scan_stations), dtype=np.float64)
Jun_ISCCP=np.zeros((Total_Month_days[5]*8, num_scan_stations), dtype=np.float64)
Jul_ISCCP=np.zeros((Total_Month_days[6]*8, num_scan_stations), dtype=np.float64)
Aug_ISCCP=np.zeros((Total_Month_days[7]*8, num_scan_stations), dtype=np.float64)
Sep_ISCCP=np.zeros((Total_Month_days[8]*8, num_scan_stations), dtype=np.float64)
Oct_ISCCP=np.zeros((Total_Month_days[9]*8, num_scan_stations), dtype=np.float64)
Nov_ISCCP=np.zeros((Total_Month_days[10]*8, num_scan_stations), dtype=np.float64)
Dec_ISCCP=np.zeros((Total_Month_days[11]*8, num_scan_stations), dtype=np.float64)
Winter_ISCCP=np.zeros((num_Winter_days*8, num_scan_stations), dtype=np.float64)
Spring_ISCCP=np.zeros((num_Spring_days*8, num_scan_stations), dtype=np.float64)
Summer_ISCCP=np.zeros((num_Summer_days*8, num_scan_stations), dtype=np.float64)
Fall_ISCCP=np.zeros((num_Fall_days*8, num_scan_stations), dtype=np.float64)

Jan_00_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_00_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_00_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_00_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_00_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_00_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_00_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_00_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_00_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_00_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_00_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_00_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_00_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_00_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_00_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_00_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_03_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_03_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_03_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_03_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_03_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_03_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_03_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_03_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_03_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_03_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_03_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_03_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_03_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_03_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_03_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_03_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_06_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_06_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_06_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_06_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_06_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_06_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_06_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_06_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_06_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_06_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_06_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_06_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_06_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_06_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_06_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_06_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_09_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_09_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_09_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_09_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_09_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_09_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_09_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_09_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_09_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_09_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_09_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_09_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_09_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_09_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_09_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_09_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_12_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_12_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_12_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_12_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_12_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_12_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_12_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_12_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_12_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_12_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_12_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_12_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_12_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_12_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_12_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_12_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_15_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_15_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_15_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_15_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_15_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_15_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_15_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_15_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_15_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_15_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_15_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_15_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_15_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_15_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_15_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_15_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_18_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_18_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_18_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_18_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_18_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_18_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_18_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_18_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_18_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_18_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_18_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_18_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_18_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_18_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_18_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_18_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

Jan_21_ISCCP=np.zeros((Total_Month_days[0], num_scan_stations), dtype=np.float64)
Feb_21_ISCCP=np.zeros((Total_Month_days[1], num_scan_stations), dtype=np.float64)
Mar_21_ISCCP=np.zeros((Total_Month_days[2], num_scan_stations), dtype=np.float64)
Apr_21_ISCCP=np.zeros((Total_Month_days[3], num_scan_stations), dtype=np.float64)
May_21_ISCCP=np.zeros((Total_Month_days[4], num_scan_stations), dtype=np.float64)
Jun_21_ISCCP=np.zeros((Total_Month_days[5], num_scan_stations), dtype=np.float64)
Jul_21_ISCCP=np.zeros((Total_Month_days[6], num_scan_stations), dtype=np.float64)
Aug_21_ISCCP=np.zeros((Total_Month_days[7], num_scan_stations), dtype=np.float64)
Sep_21_ISCCP=np.zeros((Total_Month_days[8], num_scan_stations), dtype=np.float64)
Oct_21_ISCCP=np.zeros((Total_Month_days[9], num_scan_stations), dtype=np.float64)
Nov_21_ISCCP=np.zeros((Total_Month_days[10], num_scan_stations), dtype=np.float64)
Dec_21_ISCCP=np.zeros((Total_Month_days[11], num_scan_stations), dtype=np.float64)
Winter_21_ISCCP=np.zeros((num_Winter_days, num_scan_stations), dtype=np.float64)
Spring_21_ISCCP=np.zeros((num_Spring_days, num_scan_stations), dtype=np.float64)
Summer_21_ISCCP=np.zeros((num_Summer_days, num_scan_stations), dtype=np.float64)
Fall_21_ISCCP=np.zeros((num_Fall_days, num_scan_stations), dtype=np.float64)

points=0
curr_dtg=beg_DTG
while curr_dtg <= end_DTG:

    The_YYYY=curr_dtg.year
    The_MM=curr_dtg.month
    The_DD=curr_dtg.day
    The_HH=curr_dtg.hour

    The_YYYY_str=str(The_YYYY)
    The_MM_str=str(The_MM)
    The_DD_str=str(The_DD)
    The_HH_str=str(The_HH)

    if points == 0:
        dtg_array=["%s/%s/%s %s:00" % (The_YYYY_str,The_MM_str,The_DD_str, The_HH_str)]
    else:
        dtg_array=dtg_array+["%s/%s/%s %s:00" % (The_YYYY_str,The_MM_str,The_DD_str, The_HH_str)]

    curr_dtg=curr_dtg+datetime.timedelta(hours=3)
    the_pts_array[points]=points

    if The_HH == 0 :
        print ('error checking in HH=0', the_zero_hour_csv_index, points, SCAN_zero_array_soiltemp.shape, SCAN_DATA_ARRAY.shape)
        SCAN_zero_array_soiltemp[:,the_zero_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_zero_array_soiltemp[the_zero_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_zero_hour_csv_index+=1
        dtg_array_zero=dtg_array_zero+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 3:
        
        SCAN_three_array_soiltemp[:,the_three_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_three_array_soiltemp[the_three_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_three_hour_csv_index+=1
        dtg_array_three=dtg_array_three+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 6:
        
        SCAN_six_array_soiltemp[:,the_six_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_six_array_soiltemp[the_six_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_six_hour_csv_index+=1
        dtg_array_six=dtg_array_six+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 9:
        
        SCAN_nine_array_soiltemp[:,the_nine_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_nine_array_soiltemp[the_nine_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_nine_hour_csv_index+=1
        dtg_array_nine=dtg_array_nine+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 12:
        
        SCAN_twelve_array_soiltemp[:,the_twlv_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_twelve_array_soiltemp[the_twlv_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_twlv_hour_csv_index+=1
        dtg_array_twelve=dtg_array_twelve+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 15:
        
        SCAN_fifteen_array_soiltemp[:,the_fftn_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_fifteen_array_soiltemp[the_fftn_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_fftn_hour_csv_index+=1
        dtg_array_fifteen=dtg_array_fifteen+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 18:
        
        SCAN_eighteen_array_soiltemp[:,the_eithn_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_eighteen_array_soiltemp[the_eithn_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_eithn_hour_csv_index+=1
        dtg_array_eighteen=dtg_array_eighteen+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]

    elif The_HH == 21:
        
        SCAN_twtyone_array_soiltemp[:,the_twtone_hour_csv_index,:]=SCAN_DATA_ARRAY[:,points,:]
        ISSCP_twtyone_array_soiltemp[the_twtone_hour_csv_index,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        the_twtone_hour_csv_index+=1
        dtg_array_twtyone=dtg_array_twtyone+["%s/%s/%s" % (The_YYYY_str,The_MM_str,The_DD_str)]



    if The_MM == 1:
        Jan_SCAN[:,Jan_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Jan_ISCCP[Jan_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            
            Jan_00_SCAN[:,Jan_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_00_ISCCP[Jan_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            
            Jan_00_Inc+=1
            
        if The_HH == 3:
            Jan_03_SCAN[:,Jan_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_03_ISCCP[Jan_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_03_Inc+=1
            
        if The_HH == 6:
            Jan_06_SCAN[:,Jan_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_06_ISCCP[Jan_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_06_Inc+=1
            
        if The_HH == 9:
            Jan_09_SCAN[:,Jan_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_09_ISCCP[Jan_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_09_Inc+=1
            
        if The_HH == 12:
            Jan_12_SCAN[:,Jan_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_12_ISCCP[Jan_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_12_Inc+=1
            
        if The_HH == 15:
            Jan_15_SCAN[:,Jan_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_15_ISCCP[Jan_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_15_Inc+=1
            
        if The_HH == 18:
            Jan_18_SCAN[:,Jan_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_18_ISCCP[Jan_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_18_Inc+=1
            
        if The_HH == 21:
            Jan_21_SCAN[:,Jan_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jan_21_ISCCP[Jan_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jan_21_Inc+=1
        
        Jan_Inc+=1
        
    if The_MM == 2:
        Feb_SCAN[:,Feb_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Feb_ISCCP[Feb_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        if The_HH == 0:
        
            Feb_00_SCAN[:,Feb_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_00_ISCCP[Feb_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_00_Inc+=1
            
        if The_HH == 3:
            Feb_03_SCAN[:,Feb_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_03_ISCCP[Feb_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_03_Inc+=1
            
        if The_HH == 6:
            Feb_06_SCAN[:,Feb_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_06_ISCCP[Feb_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_06_Inc+=1
            
        if The_HH == 9:
            Feb_09_SCAN[:,Feb_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_09_ISCCP[Feb_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_09_Inc+=1
            
        if The_HH == 12:
            Feb_12_SCAN[:,Feb_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_12_ISCCP[Feb_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_12_Inc+=1
            
        if The_HH == 15:
            Feb_15_SCAN[:,Feb_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_15_ISCCP[Feb_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_15_Inc+=1
            
        if The_HH == 18:
            Feb_18_SCAN[:,Feb_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_18_ISCCP[Feb_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_18_Inc+=1
            
        if The_HH == 21:
            Feb_21_SCAN[:,Feb_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Feb_21_ISCCP[Feb_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Feb_21_Inc+=1
        Feb_Inc+=1
        
    if The_MM == 3:
        Mar_SCAN[:,Mar_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Mar_ISCCP[Mar_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Mar_00_SCAN[:,Mar_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_00_ISCCP[Mar_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_00_Inc+=1
            
        if The_HH == 3:
            Mar_03_SCAN[:,Mar_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_03_ISCCP[Mar_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_03_Inc+=1
            
        if The_HH == 6:
            Mar_06_SCAN[:,Mar_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_06_ISCCP[Mar_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_06_Inc+=1
            
        if The_HH == 9:
            Mar_09_SCAN[:,Mar_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_09_ISCCP[Mar_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_09_Inc+=1
            
        if The_HH == 12:
            Mar_12_SCAN[:,Mar_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_12_ISCCP[Mar_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_12_Inc+=1
            
        if The_HH == 15:
            Mar_15_SCAN[:,Mar_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_15_ISCCP[Mar_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_15_Inc+=1
            
        if The_HH == 18:
            Mar_18_SCAN[:,Mar_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_18_ISCCP[Mar_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_18_Inc+=1
            
        if The_HH == 21:
            Mar_21_SCAN[:,Mar_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Mar_21_ISCCP[Mar_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Mar_21_Inc+=1
            
        Mar_Inc+=1
        
    if The_MM == 4:
        Apr_SCAN[:,Apr_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Apr_ISCCP[Apr_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Apr_00_SCAN[:,Apr_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_00_ISCCP[Apr_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_00_Inc+=1
            
        if The_HH == 3:
            Apr_03_SCAN[:,Apr_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_03_ISCCP[Apr_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_03_Inc+=1
            
        if The_HH == 6:
            Apr_06_SCAN[:,Apr_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_06_ISCCP[Apr_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_06_Inc+=1
            
        if The_HH == 9:
            Apr_09_SCAN[:,Apr_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_09_ISCCP[Apr_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_09_Inc+=1
            
        if The_HH == 12:
            Apr_12_SCAN[:,Apr_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_12_ISCCP[Apr_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_12_Inc+=1
            
        if The_HH == 15:
            Apr_15_SCAN[:,Apr_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_15_ISCCP[Apr_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_15_Inc+=1
            
        if The_HH == 18:
            Apr_18_SCAN[:,Apr_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_18_ISCCP[Apr_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_18_Inc+=1
            
        if The_HH == 21:
            Apr_21_SCAN[:,Apr_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Apr_21_ISCCP[Apr_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Apr_21_Inc+=1
            
        Apr_Inc+=1
        
    if The_MM == 5:
        May_SCAN[:,May_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        May_ISCCP[May_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            May_00_SCAN[:,May_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_00_ISCCP[May_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_00_Inc+=1
            
        if The_HH == 3:
            May_03_SCAN[:,May_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_03_ISCCP[May_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_03_Inc+=1
            
        if The_HH == 6:
            May_06_SCAN[:,May_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_06_ISCCP[May_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_06_Inc+=1
            
        if The_HH == 9:
            May_09_SCAN[:,May_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_09_ISCCP[May_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_09_Inc+=1
            
        if The_HH == 12:
            May_12_SCAN[:,May_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_12_ISCCP[May_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_12_Inc+=1
            
        if The_HH == 15:
            May_15_SCAN[:,May_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_15_ISCCP[May_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_15_Inc+=1
            
        if The_HH == 18:
            May_18_SCAN[:,May_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_18_ISCCP[May_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_18_Inc+=1
            
        if The_HH == 21:
            May_21_SCAN[:,May_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            May_21_ISCCP[May_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            May_21_Inc+=1
            
        May_Inc+=1
        
    if The_MM == 6:
        Jun_SCAN[:,Jun_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Jun_ISCCP[Jun_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Jun_00_SCAN[:,Jun_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_00_ISCCP[Jun_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_00_Inc+=1
            
        if The_HH == 3:
            Jun_03_SCAN[:,Jun_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_03_ISCCP[Jun_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_03_Inc+=1
            
        if The_HH == 6:
            Jun_06_SCAN[:,Jun_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_06_ISCCP[Jun_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_06_Inc+=1
            
        if The_HH == 9:
            Jun_09_SCAN[:,Jun_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_09_ISCCP[Jun_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_09_Inc+=1
            
        if The_HH == 12:
            Jun_12_SCAN[:,Jun_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_12_ISCCP[Jun_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_12_Inc+=1
            
        if The_HH == 15:
            Jun_15_SCAN[:,Jun_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_15_ISCCP[Jun_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_15_Inc+=1
            
        if The_HH == 18:
            Jun_18_SCAN[:,Jun_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_18_ISCCP[Jun_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_18_Inc+=1
            
        if The_HH == 21:
            Jun_21_SCAN[:,Jun_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jun_21_ISCCP[Jun_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jun_21_Inc+=1
            
        Jun_Inc+=1
        
    if The_MM == 7:
        Jul_SCAN[:,Jul_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Jul_ISCCP[Jul_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Jul_00_SCAN[:,Jul_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_00_ISCCP[Jul_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_00_Inc+=1
            
        if The_HH == 3:
            Jul_03_SCAN[:,Jul_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_03_ISCCP[Jul_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_03_Inc+=1
            
        if The_HH == 6:
            Jul_06_SCAN[:,Jul_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_06_ISCCP[Jul_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_06_Inc+=1
            
        if The_HH == 9:
            Jul_09_SCAN[:,Jul_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_09_ISCCP[Jul_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_09_Inc+=1
            
        if The_HH == 12:
            Jul_12_SCAN[:,Jul_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_12_ISCCP[Jul_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_12_Inc+=1
            
        if The_HH == 15:
            Jul_15_SCAN[:,Jul_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_15_ISCCP[Jul_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_15_Inc+=1
            
        if The_HH == 18:
            Jul_18_SCAN[:,Jul_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_18_ISCCP[Jul_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_18_Inc+=1
            
        if The_HH == 21:
            Jul_21_SCAN[:,Jul_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Jul_21_ISCCP[Jul_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Jul_21_Inc+=1
            
        Jul_Inc+=1
        
    if The_MM == 8:
        Aug_SCAN[:,Aug_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Aug_ISCCP[Aug_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Aug_00_SCAN[:,Aug_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_00_ISCCP[Aug_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_00_Inc+=1
            
        if The_HH == 3:
            Aug_03_SCAN[:,Aug_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_03_ISCCP[Aug_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_03_Inc+=1
            
        if The_HH == 6:
            Aug_06_SCAN[:,Aug_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_06_ISCCP[Aug_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_06_Inc+=1
            
        if The_HH == 9:
            Aug_09_SCAN[:,Aug_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_09_ISCCP[Aug_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_09_Inc+=1
            
        if The_HH == 12:
            Aug_12_SCAN[:,Aug_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_12_ISCCP[Aug_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_12_Inc+=1
            
        if The_HH == 15:
            Aug_15_SCAN[:,Aug_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_15_ISCCP[Aug_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_15_Inc+=1
            
        if The_HH == 18:
            Aug_18_SCAN[:,Aug_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_18_ISCCP[Aug_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_18_Inc+=1
            
        if The_HH == 21:
            Aug_21_SCAN[:,Aug_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Aug_21_ISCCP[Aug_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Aug_21_Inc+=1
            
        Aug_Inc+=1
        
    if The_MM == 9:
        Sep_SCAN[:,Sep_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Sep_ISCCP[Sep_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Sep_00_SCAN[:,Sep_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_00_ISCCP[Sep_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_00_Inc+=1
            
        if The_HH == 3:
            Sep_03_SCAN[:,Sep_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_03_ISCCP[Sep_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_03_Inc+=1
            
        if The_HH == 6:
            Sep_06_SCAN[:,Sep_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_06_ISCCP[Sep_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_06_Inc+=1
            
        if The_HH == 9:
            Sep_09_SCAN[:,Sep_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_09_ISCCP[Sep_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_09_Inc+=1
            
        if The_HH == 12:
            Sep_12_SCAN[:,Sep_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_12_ISCCP[Sep_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_12_Inc+=1
            
        if The_HH == 15:
            Sep_15_SCAN[:,Sep_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_15_ISCCP[Sep_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_15_Inc+=1
            
        if The_HH == 18:
            Sep_18_SCAN[:,Sep_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_18_ISCCP[Sep_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_18_Inc+=1
            
        if The_HH == 21:
            Sep_21_SCAN[:,Sep_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Sep_21_ISCCP[Sep_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Sep_21_Inc+=1
            
        Sep_Inc+=1
        
    if The_MM == 10:
        Oct_SCAN[:,Oct_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Oct_ISCCP[Oct_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Oct_00_SCAN[:,Oct_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_00_ISCCP[Oct_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_00_Inc+=1
            
        if The_HH == 3:
            Oct_03_SCAN[:,Oct_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_03_ISCCP[Oct_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_03_Inc+=1
            
        if The_HH == 6:
            Oct_06_SCAN[:,Oct_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_06_ISCCP[Oct_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_06_Inc+=1
            
        if The_HH == 9:
            Oct_09_SCAN[:,Oct_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_09_ISCCP[Oct_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_09_Inc+=1
            
        if The_HH == 12:
            Oct_12_SCAN[:,Oct_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_12_ISCCP[Oct_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_12_Inc+=1
            
        if The_HH == 15:
            Oct_15_SCAN[:,Oct_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_15_ISCCP[Oct_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_15_Inc+=1
            
        if The_HH == 18:
            Oct_18_SCAN[:,Oct_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_18_ISCCP[Oct_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_18_Inc+=1
            
        if The_HH == 21:
            Oct_21_SCAN[:,Oct_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Oct_21_ISCCP[Oct_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Oct_21_Inc+=1
            
        Oct_Inc+=1
        
    if The_MM == 11:
        Nov_SCAN[:,Nov_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Nov_ISCCP[Nov_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Nov_00_SCAN[:,Nov_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_00_ISCCP[Nov_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_00_Inc+=1
            
        if The_HH == 3:
            Nov_03_SCAN[:,Nov_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_03_ISCCP[Nov_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_03_Inc+=1
            
        if The_HH == 6:
            Nov_06_SCAN[:,Nov_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_06_ISCCP[Nov_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_06_Inc+=1
            
        if The_HH == 9:
            Nov_09_SCAN[:,Nov_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_09_ISCCP[Nov_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_09_Inc+=1
            
        if The_HH == 12:
            Nov_12_SCAN[:,Nov_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_12_ISCCP[Nov_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_12_Inc+=1
            
        if The_HH == 15:
            Nov_15_SCAN[:,Nov_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_15_ISCCP[Nov_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_15_Inc+=1
            
        if The_HH == 18:
            Nov_18_SCAN[:,Nov_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_18_ISCCP[Nov_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_18_Inc+=1
            
        if The_HH == 21:
            Nov_21_SCAN[:,Nov_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Nov_21_ISCCP[Nov_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Nov_21_Inc+=1
            
        Nov_Inc+=1
        
    if The_MM == 12:
        Dec_SCAN[:,Dec_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
        Dec_ISCCP[Dec_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
        
        if The_HH == 0:
            Dec_00_SCAN[:,Dec_00_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_00_ISCCP[Dec_00_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_00_Inc+=1
            
        if The_HH == 3:
            Dec_03_SCAN[:,Dec_03_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_03_ISCCP[Dec_03_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_03_Inc+=1
            
        if The_HH == 6:
            Dec_06_SCAN[:,Dec_06_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_06_ISCCP[Dec_06_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_06_Inc+=1
            
        if The_HH == 9:
            Dec_09_SCAN[:,Dec_09_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_09_ISCCP[Dec_09_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_09_Inc+=1
            
        if The_HH == 12:
            Dec_12_SCAN[:,Dec_12_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_12_ISCCP[Dec_12_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_12_Inc+=1
            
        if The_HH == 15:
            Dec_15_SCAN[:,Dec_15_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_15_ISCCP[Dec_15_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_15_Inc+=1
            
        if The_HH == 18:
            Dec_18_SCAN[:,Dec_18_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_18_ISCCP[Dec_18_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_18_Inc+=1
            
        if The_HH == 21:
            Dec_21_SCAN[:,Dec_21_Inc,:]=SCAN_DATA_ARRAY[:,points,:]
            Dec_21_ISCCP[Dec_21_Inc,:]=ISCCP_TSKIN_ARRAY[points,:]
            Dec_21_Inc+=1

    points+=1
    
Winter_00_SCAN=np.concatenate((Dec_00_SCAN, Jan_00_SCAN, Feb_00_SCAN), axis=1)
Winter_03_SCAN=np.concatenate((Dec_03_SCAN, Jan_03_SCAN, Feb_03_SCAN), axis=1)
Winter_06_SCAN=np.concatenate((Dec_06_SCAN, Jan_06_SCAN, Feb_06_SCAN), axis=1)
Winter_09_SCAN=np.concatenate((Dec_09_SCAN, Jan_09_SCAN, Feb_09_SCAN), axis=1)
Winter_12_SCAN=np.concatenate((Dec_12_SCAN, Jan_12_SCAN, Feb_12_SCAN), axis=1)
Winter_15_SCAN=np.concatenate((Dec_15_SCAN, Jan_15_SCAN, Feb_15_SCAN), axis=1)
Winter_18_SCAN=np.concatenate((Dec_18_SCAN, Jan_18_SCAN, Feb_18_SCAN), axis=1)
Winter_21_SCAN=np.concatenate((Dec_21_SCAN, Jan_21_SCAN, Feb_21_SCAN), axis=1)

Winter_00_ISCCP=np.concatenate((Dec_00_ISCCP, Jan_00_ISCCP, Feb_00_ISCCP), axis=0)
Winter_03_ISCCP=np.concatenate((Dec_03_ISCCP, Jan_03_ISCCP, Feb_03_ISCCP), axis=0)
Winter_06_ISCCP=np.concatenate((Dec_06_ISCCP, Jan_06_ISCCP, Feb_06_ISCCP), axis=0)
Winter_09_ISCCP=np.concatenate((Dec_09_ISCCP, Jan_09_ISCCP, Feb_09_ISCCP), axis=0)
Winter_12_ISCCP=np.concatenate((Dec_12_ISCCP, Jan_12_ISCCP, Feb_12_ISCCP), axis=0)
Winter_15_ISCCP=np.concatenate((Dec_15_ISCCP, Jan_15_ISCCP, Feb_15_ISCCP), axis=0)
Winter_18_ISCCP=np.concatenate((Dec_18_ISCCP, Jan_18_ISCCP, Feb_18_ISCCP), axis=0)
Winter_21_ISCCP=np.concatenate((Dec_21_ISCCP, Jan_21_ISCCP, Feb_21_ISCCP), axis=0)

Spring_00_SCAN=np.concatenate((Mar_00_SCAN, Apr_00_SCAN, May_00_SCAN), axis=1)
Spring_03_SCAN=np.concatenate((Mar_03_SCAN, Apr_03_SCAN, May_03_SCAN), axis=1)
Spring_06_SCAN=np.concatenate((Mar_06_SCAN, Apr_06_SCAN, May_06_SCAN), axis=1)
Spring_09_SCAN=np.concatenate((Mar_09_SCAN, Apr_09_SCAN, May_09_SCAN), axis=1)
Spring_12_SCAN=np.concatenate((Mar_12_SCAN, Apr_12_SCAN, May_12_SCAN), axis=1)
Spring_15_SCAN=np.concatenate((Mar_15_SCAN, Apr_15_SCAN, May_15_SCAN), axis=1)
Spring_18_SCAN=np.concatenate((Mar_18_SCAN, Apr_18_SCAN, May_18_SCAN), axis=1)
Spring_21_SCAN=np.concatenate((Mar_21_SCAN, Apr_21_SCAN, May_21_SCAN), axis=1)

Spring_00_ISCCP=np.concatenate((Mar_00_ISCCP, Apr_00_ISCCP, May_00_ISCCP), axis=0)
Spring_03_ISCCP=np.concatenate((Mar_03_ISCCP, Apr_03_ISCCP, May_03_ISCCP), axis=0)
Spring_06_ISCCP=np.concatenate((Mar_06_ISCCP, Apr_06_ISCCP, May_06_ISCCP), axis=0)
Spring_09_ISCCP=np.concatenate((Mar_09_ISCCP, Apr_09_ISCCP, May_09_ISCCP), axis=0)
Spring_12_ISCCP=np.concatenate((Mar_12_ISCCP, Apr_12_ISCCP, May_12_ISCCP), axis=0)
Spring_15_ISCCP=np.concatenate((Mar_15_ISCCP, Apr_15_ISCCP, May_15_ISCCP), axis=0)
Spring_18_ISCCP=np.concatenate((Mar_18_ISCCP, Apr_18_ISCCP, May_18_ISCCP), axis=0)
Spring_21_ISCCP=np.concatenate((Mar_21_ISCCP, Apr_21_ISCCP, May_21_ISCCP), axis=0)

Summer_00_SCAN=np.concatenate((Jun_00_SCAN, Jul_00_SCAN, Aug_00_SCAN), axis=1)
Summer_03_SCAN=np.concatenate((Jun_03_SCAN, Jul_03_SCAN, Aug_03_SCAN), axis=1)
Summer_06_SCAN=np.concatenate((Jun_06_SCAN, Jul_06_SCAN, Aug_06_SCAN), axis=1)
Summer_09_SCAN=np.concatenate((Jun_09_SCAN, Jul_09_SCAN, Aug_09_SCAN), axis=1)
Summer_12_SCAN=np.concatenate((Jun_12_SCAN, Jul_12_SCAN, Aug_12_SCAN), axis=1)
Summer_15_SCAN=np.concatenate((Jun_15_SCAN, Jul_15_SCAN, Aug_15_SCAN), axis=1)
Summer_18_SCAN=np.concatenate((Jun_18_SCAN, Jul_18_SCAN, Aug_18_SCAN), axis=1)
Summer_21_SCAN=np.concatenate((Jun_21_SCAN, Jul_21_SCAN, Aug_21_SCAN), axis=1)

Summer_00_ISCCP=np.concatenate((Jun_00_ISCCP, Jul_00_ISCCP, Aug_00_ISCCP), axis=0)
Summer_03_ISCCP=np.concatenate((Jun_03_ISCCP, Jul_03_ISCCP, Aug_03_ISCCP), axis=0)
Summer_06_ISCCP=np.concatenate((Jun_06_ISCCP, Jul_06_ISCCP, Aug_06_ISCCP), axis=0)
Summer_09_ISCCP=np.concatenate((Jun_09_ISCCP, Jul_09_ISCCP, Aug_09_ISCCP), axis=0)
Summer_12_ISCCP=np.concatenate((Jun_12_ISCCP, Jul_12_ISCCP, Aug_12_ISCCP), axis=0)
Summer_15_ISCCP=np.concatenate((Jun_15_ISCCP, Jul_15_ISCCP, Aug_15_ISCCP), axis=0)
Summer_18_ISCCP=np.concatenate((Jun_18_ISCCP, Jul_18_ISCCP, Aug_18_ISCCP), axis=0)
Summer_21_ISCCP=np.concatenate((Jun_21_ISCCP, Jul_21_ISCCP, Aug_21_ISCCP), axis=0)

Fall_00_SCAN=np.concatenate((Sep_00_SCAN, Oct_00_SCAN, Nov_00_SCAN), axis=1)
Fall_03_SCAN=np.concatenate((Sep_03_SCAN, Oct_03_SCAN, Nov_03_SCAN), axis=1)
Fall_06_SCAN=np.concatenate((Sep_06_SCAN, Oct_06_SCAN, Nov_06_SCAN), axis=1)
Fall_09_SCAN=np.concatenate((Sep_09_SCAN, Oct_09_SCAN, Nov_09_SCAN), axis=1)
Fall_12_SCAN=np.concatenate((Sep_12_SCAN, Oct_12_SCAN, Nov_12_SCAN), axis=1)
Fall_15_SCAN=np.concatenate((Sep_15_SCAN, Oct_15_SCAN, Nov_15_SCAN), axis=1)
Fall_18_SCAN=np.concatenate((Sep_18_SCAN, Oct_18_SCAN, Nov_18_SCAN), axis=1)
Fall_21_SCAN=np.concatenate((Sep_21_SCAN, Oct_21_SCAN, Nov_21_SCAN), axis=1)

Fall_00_ISCCP=np.concatenate((Sep_00_ISCCP, Oct_00_ISCCP, Nov_00_ISCCP), axis=0)
Fall_03_ISCCP=np.concatenate((Sep_03_ISCCP, Oct_03_ISCCP, Nov_03_ISCCP), axis=0)
Fall_06_ISCCP=np.concatenate((Sep_06_ISCCP, Oct_06_ISCCP, Nov_06_ISCCP), axis=0)
Fall_09_ISCCP=np.concatenate((Sep_09_ISCCP, Oct_09_ISCCP, Nov_09_ISCCP), axis=0)
Fall_12_ISCCP=np.concatenate((Sep_12_ISCCP, Oct_12_ISCCP, Nov_12_ISCCP), axis=0)
Fall_15_ISCCP=np.concatenate((Sep_15_ISCCP, Oct_15_ISCCP, Nov_15_ISCCP), axis=0)
Fall_18_ISCCP=np.concatenate((Sep_18_ISCCP, Oct_18_ISCCP, Nov_18_ISCCP), axis=0)
Fall_21_ISCCP=np.concatenate((Sep_21_ISCCP, Oct_21_ISCCP, Nov_21_ISCCP), axis=0)

index_array_cycles=np.arange(int(the_hour_csv_range/num_years)+1)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot Top Layer COMPARISON
##################################################################################################################

station_number=0
while station_number < num_scan_stations-1:
    Scat_min=240
    Scat_max=340
    
    min_cor=250
    #min_cor=275
    max_cor=340
#    filtered_x=ma.masked_outside(SCAN_DATA_ARRAY[0,0:num_scan_recs-1,station_number],min_cor,350)
#    filtered_y_36_OL=ma.masked_outside(LIS_Noah36_OL_DATA_ARRAY[0,0:num_scan_recs-1,station_number],min_cor,350)
#    filtered_y_36_DA=ma.masked_outside(LIS_Noah36_DA_DATA_ARRAY[0,0:num_scan_recs-1,station_number],min_cor,350)
#    mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
#    mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_DATA_ARRAY[0,0:num_scan_recs-1,3],mask=mask_36_OL).compressed()
#    filtered_y_final_36_OL=ma.masked_array(LIS_Noah36_OL_DATA_ARRAY[0,0:num_scan_recs-1,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_DA=ma.masked_array(LIS_Noah36_DA_DATA_ARRAY[0,0:num_scan_recs-1,station_number],mask=mask_36_DA).compressed()
#    
#    if (filtered_y_final_36_OL.shape[0] > 100):
#        slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final, filtered_y_final_36_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_OL-filtered_x_final
#        temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_36_DA.shape[0] > 100):
#        slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final, filtered_y_final_36_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_DA-filtered_x_final
#        temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
#    
#    filtered_x=ma.masked_outside(SCAN_DATA_ARRAY[0,0:num_scan_recs-1,station_number],min_cor,350)
#    filtered_y_MP_OL=ma.masked_outside(LIS_NoahMP_OL_DATA_ARRAY[0,0:num_scan_recs-1,station_number],min_cor,350)
#    filtered_y_MP_DA=ma.masked_outside(LIS_NoahMP_DA_DATA_ARRAY[0,0:num_scan_recs-1,station_number],min_cor,350)
#    mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
#    mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_DATA_ARRAY[0,0:num_scan_recs-1,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_OL=ma.masked_array(LIS_NoahMP_OL_DATA_ARRAY[0,0:num_scan_recs-1,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_DA=ma.masked_array(LIS_NoahMP_DA_DATA_ARRAY[0,0:num_scan_recs-1,station_number],mask=mask_MP_DA).compressed()
#    
#    if (filtered_y_final_MP_OL.shape[0] > 100):
#        slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final, filtered_y_final_MP_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_OL-filtered_x_final
#        temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_MP_DA.shape[0] > 100):
#        slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final, filtered_y_final_MP_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_DA-filtered_x_final
#        temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
    
    figure, axes=plt.subplots(nrows=2, figsize=(15,8))
    axes[0].set_ylim(240, 340)
    print (index_array.shape, SCAN_DATA_ARRAY.shape)
    axes[0].scatter(index_array, SCAN_DATA_ARRAY[0,0:num_scan_recs-1,station_number], marker='*', color='b', alpha=.5, label="ISMN Data")
    axes[0].scatter(index_array, ISCCP_TSKIN_ARRAY[0:num_scan_recs-1,station_number], marker='.', color='r', alpha=.5, label="ISCCP Data")
    axes[0].legend(loc='upper right', borderaxespad=0.)
    axes[0].set_xlabel('date')
    axes[0].set_ylabel('Temp (K)')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
#    axes[0].text(2500, 250, "MP OL Bias:"+str(round(temp_bias_MP_OL,1))+"  RMS: "+str(round(r_value_MP_OL,1)), ha="left", va="center", size=8, bbox=bbox_props)
#    axes[0].text(2500, 260, "MP DA Bias:"+str(round(temp_bias_MP_DA,1))+"  RMS: "+str(round(r_value_MP_DA,1)), ha="left", va="center", size=8, bbox=bbox_props)
    
    axes[1].set_ylim(240, 340)
    axes[1].scatter(index_array, SCAN_DATA_ARRAY[1,0:num_scan_recs-1,station_number], marker='*', color='b', alpha=.5, label="ISMN Data")
    axes[1].scatter(index_array, ISCCP_TSKIN_ARRAY[0:num_scan_recs-1,station_number], marker='.', color='r', alpha=.5, label="ISCCP Data")
    axes[1].legend(loc='upper right', borderaxespad=0.)
    axes[1].set_xlabel('date')
    axes[1].set_ylabel('Temp (K)')
    img_fname_pre=img_out_path+str(stationsfile['Stat_Name'][station_number])
    plt.suptitle(str(stationsfile['Stat_Name'][station_number])+' LIS Noah Open Loop vs. USDA SCAN Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
    plt.savefig(img_fname_pre+'_ISCCP_vs_ISMN_Data_year_'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
    plt.close(figure)
    station_number+=1

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot Top Layer COMPARISON
##################################################################################################################

#figure, axes=plt.subplots(nrows=2, figsize=(15,8))
#axes[0].set_ylim(240, 340)
#axes[0].plot(index_array, SCAN_DATA_ARRAY[0,0:num_scan_recs-1,3], lw=0.5)
#axes[0].plot(index_array, LIS_NoahMP_DA_DATA_ARRAY[0,0:num_scan_recs-1,3], lw=0.5)
#axes[0].set_xlabel('date')
#axes[0].set_ylabel('Temp (K)')

#axes[1].set_ylim(240, 340)
#axes[1].plot(index_array, SCAN_DATA_ARRAY[0,0:num_scan_recs-1,3], lw=0.5)
#axes[1].plot(index_array, LIS_Noah36_DA_DATA_ARRAY[0,0:num_scan_recs-1,3], lw=0.5)
#axes[1].set_xlabel('date')
#axes[1].set_ylabel('Temp (K)')
#plt.suptitle(CURR_SCANDATA.columns[3]+' LIS Noah DA Loop vs. USDA SCAN Data\n'+BGDATETXT+' - '+EDATETXT, fontsize=18)
#plt.savefig(img_out_path+'LIS_Noah_36-MP_TOP_DA_vs_USDA_SCAN_Data_year'+BGDATE+'-'+EDATE+'.png')
#plt.close(figure)

min_cor=250
#min_cor=275
max_cor=340

stations_ubrmse_array_L1=np.zeros((8, num_scan_stations-1), dtype=np.float64)
stations_bias_array_L1=np.zeros((8, num_scan_stations-1), dtype=np.float64)
stations_corr_array_L1=np.zeros((8, num_scan_stations-1), dtype=np.float64)

station_number=0
while station_number < num_scan_stations-1:
    print ('Station number =', station_number, 'number of stations total =', num_scan_stations)
    Scat_min=240
    Scat_max=340
    
    min_cor=250
    #min_cor=275
    max_cor=340
##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
##################################################################################################################
#    filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_OL=ma.masked_outside(LIS_zero_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_DA=ma.masked_outside(N36_DA_zero_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
#    mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_OL=ma.masked_array(LIS_zero_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_DA=ma.masked_array(N36_DA_zero_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
#
#    if (filtered_y_final_36_OL.shape[0] > 100):
#        slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final, filtered_y_final_36_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_OL-filtered_x_final
#        temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_36_DA.shape[0] > 100):
#        slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final, filtered_y_final_36_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_DA-filtered_x_final
#        temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
#    
#    filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_OL=ma.masked_outside(MP_zero_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_DA=ma.masked_outside(MP_DA_zero_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
#    mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_OL=ma.masked_array(MP_zero_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_DA=ma.masked_array(MP_DA_zero_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
#
#    if (filtered_y_final_MP_OL.shape[0] > 100):
#        slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final, filtered_y_final_MP_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_OL-filtered_x_final
#        temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_MP_DA.shape[0] > 100):
#        slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final, filtered_y_final_MP_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_DA-filtered_x_final
#        temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
        
    figure, axes=plt.subplots(nrows=4, figsize=(12,12))
    axes[0].set_ylim(240, 340)
    axes[0].set_xlim(0,int(the_hour_csv_range/num_years))
    axes[0].scatter(index_array_cycles, SCAN_zero_array_soiltemp[0,:,station_number], marker='*', color='b', alpha=.5, label='ISMN 5-cm')
    axes[0].scatter(index_array_cycles, ISSCP_zero_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    axes[0].grid()
    axes[0].set_ylabel('Temp (K)')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    axes[0].text(10, 330, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    axes[0].legend(loc='upper right', edgecolor="none", borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################
#    filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_OL=ma.masked_outside(LIS_six_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_DA=ma.masked_outside(N36_DA_six_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
#    mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_OL=ma.masked_array(LIS_six_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_DA=ma.masked_array(N36_DA_six_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
#
#    if (filtered_y_final_36_OL.shape[0] > 100):
#        slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final, filtered_y_final_36_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_OL-filtered_x_final
#        temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_36_DA.shape[0] > 100):
#        slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final, filtered_y_final_36_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_DA-filtered_x_final
#        temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
#    
#    filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_OL=ma.masked_outside(MP_six_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_DA=ma.masked_outside(MP_DA_six_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
#    mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_OL=ma.masked_array(MP_six_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_DA=ma.masked_array(MP_DA_six_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
#
#    if (filtered_y_final_MP_OL.shape[0] > 100):
#        slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final, filtered_y_final_MP_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_OL-filtered_x_final
#        temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_MP_DA.shape[0] > 100):
#        slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final, filtered_y_final_MP_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_DA-filtered_x_final
#        temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
        
    axes[1].set_ylim(240, 340)
    axes[1].set_xlim(0,int(the_hour_csv_range/num_years))
    axes[1].scatter(index_array_cycles, SCAN_six_array_soiltemp[0,:,station_number], marker='*', color='b', alpha=.5, label='ISMN 5-cm')
    axes[1].scatter(index_array_cycles, ISSCP_six_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    axes[1].grid()
    axes[1].set_ylabel('Temp (K)')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    axes[1].text(10, 330, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    axes[1].legend(loc='upper right', edgecolor="none", borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################

#    filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_OL=ma.masked_outside(LIS_twelve_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_DA=ma.masked_outside(N36_DA_twelve_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
#    mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_OL=ma.masked_array(LIS_twelve_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_DA=ma.masked_array(N36_DA_twelve_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
#
#    if (filtered_y_final_36_OL.shape[0] > 100):
#        slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final, filtered_y_final_36_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_OL-filtered_x_final
#        temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_36_DA.shape[0] > 100):
#        slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final, filtered_y_final_36_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_DA-filtered_x_final
#        temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
#    
#    filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_OL=ma.masked_outside(MP_twelve_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_DA=ma.masked_outside(MP_DA_twelve_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
#    mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_OL=ma.masked_array(MP_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_DA=ma.masked_array(MP_DA_twelve_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()
#
#    if (filtered_y_final_MP_OL.shape[0] > 100):
#        slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final, filtered_y_final_MP_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_OL-filtered_x_final
#        temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_MP_DA.shape[0] > 100):
#        slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final, filtered_y_final_MP_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_DA-filtered_x_final
#        temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]

    axes[2].set_ylim(240, 340)
    axes[2].set_xlim(0,int(the_hour_csv_range/num_years))
    axes[2].scatter(index_array_cycles, SCAN_twelve_array_soiltemp[0,:,station_number], marker='*', color='b', alpha=.5, label='ISMN 5-cm')
    axes[2].scatter(index_array_cycles, ISSCP_twelve_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    axes[2].grid()
    axes[2].set_ylabel('Temp (K)')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    axes[2].text(10, 330, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
    axes[2].legend(loc='upper right', edgecolor="none", borderaxespad=0.)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################
    
#    filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_OL=ma.masked_outside(LIS_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_36_DA=ma.masked_outside(N36_DA_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_36_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_OL.filled(np.nan)).mask
#    mask_36_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_36_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_OL=ma.masked_array(LIS_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_OL).compressed()
#    filtered_y_final_36_DA=ma.masked_array(N36_DA_eighteen_array_soiltemp[0,:,station_number],mask=mask_36_DA).compressed()
#
#    if (filtered_y_final_36_OL.shape[0] > 100):
#        slope_36_OL, intercept_36_OL, r_value_36_OL, p_value_36_OL, std_err_36_OL = stats.linregress(filtered_x_final, filtered_y_final_36_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_OL-filtered_x_final
#        temp_bias_36_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_36_DA.shape[0] > 100):
#        slope_36_DA, intercept_36_DA, r_value_36_DA, p_value_36_DA, std_err_36_DA = stats.linregress(filtered_x_final, filtered_y_final_36_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_36_DA-filtered_x_final
#        temp_bias_36_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
#    
#    filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_OL=ma.masked_outside(MP_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
#    filtered_y_MP_DA=ma.masked_outside(MP_DA_eighteen_array_soiltemp[0,:,station_number],min_cor,350)
#    mask_MP_OL=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_OL.filled(np.nan)).mask
#    mask_MP_DA=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y_MP_DA.filled(np.nan)).mask
#    filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_OL=ma.masked_array(MP_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_OL).compressed()
#    filtered_y_final_MP_DA=ma.masked_array(MP_DA_eighteen_array_soiltemp[0,:,station_number],mask=mask_MP_DA).compressed()

#    if (filtered_y_final_MP_OL.shape[0] > 100):
#        slope_MP_OL, intercept_MP_OL, r_value_MP_OL, p_value_MP_OL, std_err_MP_OL = stats.linregress(filtered_x_final, filtered_y_final_MP_OL)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_OL-filtered_x_final
#        temp_bias_MP_OL=np.sum(filtered_diff)/filtered_diff.shape[0]
#    if (filtered_y_final_MP_DA.shape[0] > 100):
#        slope_MP_DA, intercept_MP_DA, r_value_MP_DA, p_value_MP_DA, std_err_MP_DA = stats.linregress(filtered_x_final, filtered_y_final_MP_DA)
#        #compute the bias
#        filtered_diff=filtered_y_final_MP_DA-filtered_x_final
#        temp_bias_MP_DA=np.sum(filtered_diff)/filtered_diff.shape[0]
        
    axes[3].set_ylim(240, 340)
    axes[3].set_xlim(0,int(the_hour_csv_range/num_years))
    axes[3].scatter(index_array_cycles, SCAN_eighteen_array_soiltemp[0,:,station_number], marker='*', color='b', alpha=.5, label='ISMN 5-cm')
    axes[3].scatter(index_array_cycles, ISSCP_eighteen_array_soiltemp[:,station_number], marker='.', color='r', alpha=.5, label='ISCCP')
    axes[3].grid()
    axes[3].set_xlabel('Day of Year')
    axes[3].set_ylabel('Temp (K)')
    bbox_props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
    axes[3].text(10, 330, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)


    
    axes[3].legend(loc='upper right', edgecolor="none", borderaxespad=0.)

    plt.suptitle(str(stationsfile['Stat_Name'][station_number])+' ISCCP LST vs. ISMN Soil Temperature Data \n Valid '+BGDATETXT+' - '+EDATETXT, fontsize=18)
    #img_fname_pre=img_out_path+stationsfile['State Code'][station_number]+'-'+str(stationsfile['Stat_Name'][station_number])
    img_fname_pre=img_out_path+str(stationsfile['Stat_Name'][station_number])
    img_fname_end='-ISCCP_vs_ISMN_Data_4-Plot-year_'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png'
    plt.savefig(img_fname_pre+img_fname_end)
    plt.close(figure)


##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 1 soil temps for each station
##################################################################################################################

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ##################################################################################################################
    Scat_min=240
    Scat_max=340
    
    min_cor=250
    #min_cor=275
    max_cor=340
    
    
    figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))
    
    filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 00 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
    
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[0,0].set_ylim(Scat_min, Scat_max)
        axes[0,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[0,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[0,0].grid()
        axes[0,0].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center",size=8)
        axes[0,0].text(Scat_min-14, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-14, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[0,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)
        
        stations_ubrmse_array_L1[0, station_number]=ubrmse
        stations_bias_array_L1[0, station_number]=temp_bias
        stations_corr_array_L1[0, station_number]=r_value
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 03Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 03 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
    
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[1,0].set_ylim(Scat_min, Scat_max)
        axes[1,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[1,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[1,0].grid()
        axes[1,0].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center",size=8)
        axes[1,0].text(Scat_min-14, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-14, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[1,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)
        
        stations_ubrmse_array_L1[1, station_number]=ubrmse
        stations_bias_array_L1[1, station_number]=temp_bias
        stations_corr_array_L1[1, station_number]=r_value
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 06 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[2,0].set_ylim(Scat_min, Scat_max)
        axes[2,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[2,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[2,0].grid()
        axes[2,0].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=8)
        axes[2,0].text(Scat_min-14, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-14, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[2,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)
        
        stations_ubrmse_array_L1[2, station_number]=ubrmse
        stations_bias_array_L1[2, station_number]=temp_bias
        stations_corr_array_L1[2, station_number]=r_value
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 09Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 09 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
    
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[3,0].set_ylim(Scat_min, Scat_max)
        axes[3,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[3,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[3,0].grid()
        axes[3,0].set_ylabel('ISCCP Skin T (K)')
        axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,0].text(Scat_min-14, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-14, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-14, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=8)
        axes[3,0].text(Scat_min-14, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-14, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[3,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)
        
        stations_ubrmse_array_L1[3, station_number]=ubrmse
        stations_bias_array_L1[3, station_number]=temp_bias
        stations_corr_array_L1[3, station_number]=r_value
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 12 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)

        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[0,1].set_ylim(Scat_min, Scat_max)
        axes[0,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[0,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[0,1].grid()
        axes[0,1].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+5, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+5, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[0,1].text(Scat_max+5, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+5, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[0,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

        stations_ubrmse_array_L1[4, station_number]=ubrmse
        stations_bias_array_L1[4, station_number]=temp_bias
        stations_corr_array_L1[4, station_number]=r_value

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 15Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 15 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[1,1].set_ylim(Scat_min, Scat_max)
        axes[1,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[1,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[1,1].grid()
        axes[1,1].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+5, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+5, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[1,1].text(Scat_max+5, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+5, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[1,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

        stations_ubrmse_array_L1[5, station_number]=ubrmse
        stations_bias_array_L1[5, station_number]=temp_bias
        stations_corr_array_L1[5, station_number]=r_value

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
    
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 18 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[2,1].set_ylim(Scat_min, Scat_max)
        axes[2,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[2,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[2,1].grid()
        axes[2,1].set_ylabel('ISCCP Skin T (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,1].text(Scat_max+5, Scat_max-5,"RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+5, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+5, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[2,1].text(Scat_max+5, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+5, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[2,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

        stations_ubrmse_array_L1[6, station_number]=ubrmse
        stations_bias_array_L1[6, station_number]=temp_bias
        stations_corr_array_L1[6, station_number]=r_value
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 21Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[0,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[0,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)

        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 21 UTC ISSCP vs. ISMN LY1 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)

        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[3,1].set_ylim(Scat_min, Scat_max)
        axes[3,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[3,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
        axes[3,1].grid()
        axes[3,1].set_ylabel('ISCCP Skin T (K)')
        axes[3,1].set_xlabel('ISMN 5cm Tsoil (K)')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,1].text(Scat_max+5, Scat_max-5, "RMS :"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+5, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+5, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=8)
        axes[3,1].text(Scat_max+5, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+5, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
        pts=str(filtered_diff.shape[0])
        axes[3,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

        stations_ubrmse_array_L1[7, station_number]=ubrmse
        stations_bias_array_L1[7, station_number]=temp_bias
        stations_corr_array_L1[7, station_number]=r_value

    plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp \n  Station ID -'+str(stationsfile['Stat_Name'][station_number]), fontsize=18)
    
    img_fname_pre=img_out_path+str(stationsfile['Stat_Name'][station_number])
    plt.savefig(img_fname_pre+'_ISCCPvsISMN5cm_'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
    plt.close(figure)
    
    
##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 2 soil temps for each station
##################################################################################################################
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
    ##################################################################################################################

    figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))
    
    filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 00 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
    
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[0,0].set_ylim(Scat_min, Scat_max)
        axes[0,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[0,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,0].grid()
        axes[0,0].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,0].text(Scat_min-8, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-8, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_min-8, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
        axes[0,0].text(Scat_min-8, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[0,0].text(Scat_max-8, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
       
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 03Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 03 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[1,0].set_ylim(Scat_min, Scat_max)
        axes[1,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[1,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,0].grid()
        axes[1,0].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,0].text(Scat_min-8, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-8, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_min-8, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center",size=6)
        axes[1,0].text(Scat_min-8, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[1,0].text(Scat_max-8, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 06 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)

        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[2,0].set_ylim(Scat_min, Scat_max)
        axes[2,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[2,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,0].grid()
        axes[2,0].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,0].text(Scat_min-10, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-8, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_min-8, Scat_max-22, "(ISSCP-ISMN)",ha="right", va="center", size=6)
        axes[2,0].text(Scat_min-8, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[2,0].text(Scat_max-8, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)

    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 09Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 09 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[3,0].set_ylim(Scat_min, Scat_max)
        axes[3,0].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[3,0].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,0].grid()
        axes[3,0].set_ylabel('ISCCP Skin T (K)')
        axes[3,0].set_xlabel('SCAN 10cm Tsoil (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,0].text(Scat_min-8, Scat_max-5,  "RMS :"+str(round(rmse,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-8, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_min-8, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
        axes[3,0].text(Scat_min-8, Scat_max-35, "R2:"+str(round(r_value,2)), ha="right", va="center", size=10)
        axes[3,0].text(Scat_max-8, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="right", va="center", size=10)
       
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 12 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[0,1].set_ylim(Scat_min, Scat_max)
        axes[0,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[0,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[0,1].grid()
        axes[0,1].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
 
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 15Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
        
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 15 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)

        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[1,1].set_ylim(Scat_min, Scat_max)
        axes[1,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
        axes[1,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[1,1].grid()
        axes[1,1].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
    ##################################################################################################################

    filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)
    
        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]
        print ('this r value for the 18 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[2,1].set_ylim(Scat_min, Scat_max)
        axes[2,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

        axes[2,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[2,1].grid()
        axes[2,1].set_ylabel('ISCCP Skin T (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        
    ##################################################################################################################
    #####  PLOT THE Whole Dataset as a line plot for 21Z Top Layer COMPARISON
    ##################################################################################################################
    
    filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[1,:,station_number],min_cor,max_cor)
    filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,station_number],min_cor,max_cor)
    mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
    filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[1,:,station_number],mask=mask).compressed()
    filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,station_number],mask=mask).compressed()

    if (filtered_y_final.shape[0] > 100):
        slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_final, filtered_y_final)

        #compute the bias
        filtered_diff=filtered_y_final-filtered_x_final
        temp_bias=np.sum(filtered_diff)/filtered_diff.shape[0]

        print ('this r value for the 21 UTC ISSCP vs. ISMN LY2 '+str(stationsfile['Stat_Name'][station_number])+' is...', r_value)
        rmse = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse)
        ubrmse=np.sqrt(rmse**2-temp_bias**2)
        
        axes[3,1].set_ylim(Scat_min, Scat_max)
        axes[3,1].set_xlim(Scat_min, Scat_max)
        xy = np.vstack([filtered_x_final, filtered_y_final])
        z = gaussian_kde(xy)(xy)
        axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

        axes[3,1].plot(filtered_x_final, filtered_x_final*slope+intercept, color='red', linewidth=2)
        axes[3,1].grid()
        axes[3,1].set_ylabel('ISCCP Skin T (K)')
        axes[3,1].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
        axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
        axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value,2)), ha="left", va="center", size=10)
        axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse,2)), ha="left", va="center", size=10)
        
    plt.suptitle('ISCCP Skin Temperature vs. SCAN 10cm Soil Temp\n'+Plot_Labels_full, fontsize=18)
    
    #img_fname_pre=img_out_path+stationsfile['State Code'][station_number]+'-'+str(stationsfile['Stat_Name'][station_number])
    img_fname_pre=img_out_path+str(stationsfile['Stat_Name'][station_number])
    plt.savefig(img_fname_pre+'_ISCCPvsISMN10cm_'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
    plt.close(figure)
    station_number+=1

DF_ISCCPvsISMN_LY1=pandas.DataFrame(stations_ubrmse_array_L1, columns=list(stationsfile['Stat_Name']))
DF_ISCCPvsISMN_LY1.to_csv(img_out_path+'/'+'Station_Stats_UBRMSE_ISCCPvsISMN5cm_'+BGDATE+'-'+EDATE+'.csv')

DF_ISCCPvsISMN_LY1=pandas.DataFrame(stations_bias_array_L1, columns=list(stationsfile['Stat_Name']))
DF_ISCCPvsISMN_LY1.to_csv(img_out_path+'/'+'Station_Stats_BIAS_ISCCPvsISMN5cm_'+BGDATE+'-'+EDATE+'.csv')

DF_ISCCPvsISMN_LY1=pandas.DataFrame(stations_corr_array_L1, columns=list(stationsfile['Stat_Name']))
DF_ISCCPvsISMN_LY1.to_csv(img_out_path+'/'+'Station_Stats_CORR_ISCCPvsISMN5cm_'+BGDATE+'-'+EDATE+'.csv')

####
####Plot the stations statistics
####

figure, plt.figure(figsize=(16,8))
#axes=plt.subplots(nrows=3, ncols=2, figsize=(16,8), gridspec_kw={'width_ratios': [3, 1]})
gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], hspace=0.05)

stat_index_array=np.arange(num_scan_stations-1)

ax0=plt.subplot(gs[0,0])

ax0.grid()
ax0.set_ylim(-15, 35)
ax0.set_xlim(-1, 120)
ax0.plot([0,120], [0,0], color='black', linewidth=2)
ax0.scatter(stat_index_array+1, stations_bias_array_L1[0,:], marker='*', color='b', alpha=.5, label="00HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[1,:], marker='.', color='r', alpha=.5, label="03HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[2,:], marker='1', color='g', alpha=.5, label="06HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[3,:], marker='P', color='black', alpha=.5, label="09HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[4,:], marker='x', color='cyan', alpha=.5, label="12HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[5,:], marker='s', color='violet', alpha=.5, label="15HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[6,:], marker='v', color='b', alpha=.5, label="18HR")
ax0.scatter(stat_index_array+1, stations_bias_array_L1[7,:], marker='d', color='g', alpha=.5, label="21HR")
ax0.set_ylabel('Bias', size=12)
ax0.set_xticklabels([])
ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)


ax1=plt.subplot(gs[1,0])
ax1.grid()
ax1.set_ylim(0.5, 1.0)
ax1.set_xlim(-1, 120)
ax1.scatter(stat_index_array+1, stations_corr_array_L1[0,:], marker='*', color='b', alpha=.5, label="00HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[1,:], marker='.', color='r', alpha=.5, label="03HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[2,:], marker='1', color='g', alpha=.5, label="06HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[3,:], marker='P', color='black', alpha=.5, label="09HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[4,:], marker='x', color='cyan', alpha=.5, label="12HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[5,:], marker='s', color='violet', alpha=.5, label="15HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[6,:], marker='v', color='b', alpha=.5, label="18HR")
ax1.scatter(stat_index_array+1, stations_corr_array_L1[7,:], marker='d', color='g', alpha=.5, label="21HR")
ax1.set_ylabel('Correlation', size=12)
ax1.set_xticklabels([])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

ax2=plt.subplot(gs[2,0])
ax2.grid()
ax2.set_ylim(0, 15)
ax2.set_xlim(-1, 120)
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[0,:], marker='*', color='b', alpha=.5, label="00HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[1,:], marker='.', color='r', alpha=.5, label="03HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[2,:], marker='1', color='g', alpha=.5, label="06HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[3,:], marker='P', color='black', alpha=.5, label="09HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[4,:], marker='x', color='cyan', alpha=.5, label="12HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[5,:], marker='s', color='violet', alpha=.5, label="15HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[6,:], marker='v', color='b', alpha=.5, label="18HR")
ax2.scatter(stat_index_array+1, stations_ubrmse_array_L1[7,:], marker='d', color='g', alpha=.5, label="21HR")
ax2.set_ylabel('ubRMSE', size=12)
ax2.set_xlabel('Station Number', size=12)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)


ax3=plt.subplot(gs[:,1])
ax3.plot([0,0], [1,1], marker='None', color='b', alpha=.5, label="00HR")
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_ylim(0, 110)
ax3.set_xlim(0,20)
ax3.axis('off')
x_loc=5
y_loc=0
station_number=0
while station_number < num_scan_stations-1:
    if (station_number < (num_scan_stations / 2.0)):
        ax3.text(x_loc, y_loc, str(station_number+1)+"-"+stationsfile['Stat_Name'][station_number], ha="left", va="center", size=8)
        old_y_loc=y_loc
    else:
        ax3.text(x_loc+10, y_loc-old_y_loc, str(station_number+1)+"-"+stationsfile['Stat_Name'][station_number], ha="left", va="center", size=8)
    
    y_loc = y_loc + 2
    station_number+=1

figure.subplots_adjust(hspace=0.05)
        
plt.suptitle('ISCCP LST vs. ISMN 5-cm Soil Temperature Station Statistics \n'+BGDATE+'-'+EDATE, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_fname_pre+'/'+'Station_Stats_BIAS_ISCCPvsISMN5cm_'+BGDATE+'-'+EDATE+'.png')
plt.close(figure)
    
##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 1 soil temps for each station
##################################################################################################################


figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))

filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,:],mask=mask).compressed()
slope_00, intercept_00, r_value_00, p_value_00, std_err_00 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 00 UTC ISSCP vs. ISMN LY1 is...', r_value_00)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_00=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_00 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_00)
ubrmse_00=np.sqrt(rmse_00**2-temp_bias_00**2)

axes[0,0].set_ylim(Scat_min, Scat_max)
axes[0,0].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,0].plot(filtered_x_final, filtered_x_final*slope_00+intercept_00, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_00,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_00,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_00,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_00,2)), ha="right", va="center", size=10)


##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 03Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,:],mask=mask).compressed()
slope_03, intercept_03, r_value_03, p_value_03, std_err_03 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 03 UTC ISSCP vs. ISMN LY1 is...', r_value_03)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_03=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_03 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_03)
ubrmse_03=np.sqrt(rmse_03**2-temp_bias_03**2)

axes[1,0].set_ylim(Scat_min, Scat_max)
axes[1,0].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,0].plot(filtered_x_final, filtered_x_final*slope_03+intercept_03, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_03,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_03,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_03,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_03,2)), ha="right", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,:],mask=mask).compressed()
slope_06, intercept_06, r_value_06, p_value_06, std_err_06 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 06 UTC ISSCP vs. ISMN LY1 is...', r_value_06)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_06=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_06 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_06)
ubrmse_06=np.sqrt(rmse_06**2-temp_bias_06**2)

axes[2,0].set_ylim(Scat_min, Scat_max)
axes[2,0].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,0].plot(filtered_x_final, filtered_x_final*slope_06+intercept_06, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_06,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_06,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_06,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_06,2)), ha="right", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 09Z Top Layer COMPARISON
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,:],mask=mask).compressed()
slope_09, intercept_09, r_value_09, p_value_09, std_err_09 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 09 UTC ISSCP vs. ISMN LY1 is...', r_value_09)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_09=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_09 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_09)
ubrmse_09=np.sqrt(rmse_09**2-temp_bias_09**2)

axes[3,0].set_ylim(Scat_min, Scat_max)
axes[3,0].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,0].plot(filtered_x_final, filtered_x_final*slope_09+intercept_09, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_09,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_09,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_09,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_09,2)), ha="right", va="center", size=10)
    
##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,:],mask=mask).compressed()
slope_12, intercept_12, r_value_12, p_value_12, std_err_12 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 12 UTC ISSCP vs. ISMN LY1 is...', r_value_12)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_12=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_12 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_12)
ubrmse_12=np.sqrt(rmse_12**2-temp_bias_12**2)

axes[0,1].set_ylim(Scat_min, Scat_max)
axes[0,1].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,1].plot(filtered_x_final, filtered_x_final*slope_12+intercept_12, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_12,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_12,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_12,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_12,2)), ha="left", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 15Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,:],mask=mask).compressed()
slope_15, intercept_15, r_value_15, p_value_15, std_err_15 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 15 UTC ISSCP vs. ISMN LY1 is...', r_value_15)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_15=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_15 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_15)
ubrmse_15=np.sqrt(rmse_15**2-temp_bias_15**2)

axes[1,1].set_ylim(Scat_min, Scat_max)
axes[1,1].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,1].plot(filtered_x_final, filtered_x_final*slope_15+intercept_15, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_15,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_15,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_15,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_15,2)), ha="left", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,:],mask=mask).compressed()
slope_18, intercept_18, r_value_18, p_value_18, std_err_18 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 18 UTC ISSCP vs. ISMN LY1 is...', r_value_18)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_18=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_18 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_18)
ubrmse_18=np.sqrt(rmse_18**2-temp_bias_18**2)

axes[2,1].set_ylim(Scat_min, Scat_max)
axes[2,1].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,1].plot(filtered_x_final, filtered_x_final*slope_18+intercept_18, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_18,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_18,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=8)
axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_18,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_18,2)), ha="left", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 21Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,:],mask=mask).compressed()
slope_21, intercept_21, r_value_21, p_value_21, std_err_21 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 21 UTC ISSCP vs. ISMN LY1 is...', r_value_18)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_21=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_21 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_21)
ubrmse_21=np.sqrt(rmse_21**2-temp_bias_21**2)

axes[3,1].set_ylim(Scat_min, Scat_max)
axes[3,1].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,1].plot(filtered_x_final, filtered_x_final*slope_21+intercept_21, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 5 cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_21,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_21,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=8)
axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_21,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_21,2)), ha="left", va="center", size=10)
    
plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp\n'+Plot_Labels_full, fontsize=18)

#img_fname_pre=img_out_path+stationsfile['State Code'][station_number]+'-'+str(stationsfile['Stat_Name'][station_number])
img_fname_pre=img_out_path
plt.savefig(img_out_path+'ISMN5cm_All_Stations_'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)

##################################################################################################################
#####  Create a plot of all the Bias, RMSE, and Correlations
##################################################################################################################

figure, axes=plt.subplots(nrows=4, ncols=1, figsize=(8,8))

Bias_array=np.array((temp_bias_00, temp_bias_03, temp_bias_06, temp_bias_09, temp_bias_12, temp_bias_15, temp_bias_18, temp_bias_21))
CORR_array=np.array((r_value_00, r_value_03, r_value_06, r_value_09, r_value_12, r_value_15, r_value_18, r_value_21))
RMSE_array=np.array((rmse_00, rmse_03, rmse_06, rmse_09, rmse_12, rmse_15, rmse_18, rmse_21))
ubRMSE_array=np.array((ubrmse_00, ubrmse_03, ubrmse_06, ubrmse_09, ubrmse_12, ubrmse_15, ubrmse_18, ubrmse_21))

x_array=np.array((1,2,3,4,5,6,7,8))

axes[0].set_ylim(-20, 20)
axes[0].set_xlim(0, 9)
axes[0].grid(which='major', axis='y')
axes[0].set_xticklabels([])
axes[0].plot(x_array, Bias_array, color='red', linewidth=2)
axes[0].set_ylabel('Bias')


axes[1].set_ylim(-1, 1)
axes[1].set_xlim(0, 9)
axes[1].grid(which='major', axis='y')
axes[1].set_xticks([0,1,2,3,4,5,6,7,8])
axes[1].set_xticklabels([])
axes[1].plot(x_array, CORR_array, color='blue', linewidth=2)
axes[1].set_ylabel('Correlation')

axes[2].set_ylim(-10, 10)
axes[2].set_xlim(0, 9)
axes[2].grid(which='major', axis='y')
axes[2].set_xticks([0,1,2,3,4,5,6,7,8])
axes[2].set_xticklabels([])
axes[2].plot(x_array, RMSE_array, color='green', linewidth=2)
axes[2].set_ylabel('RMSE')
plt.subplots_adjust(wspace=None, hspace=None)

axes[3].set_ylim(-10, 10)
axes[3].set_xlim(0, 9)
axes[3].grid(which='major', axis='y')
axes[3].set_xticks([0,1,2,3,4,5,6,7,8])
axes[3].set_xticklabels([' ', '00', '03', '06', '09', '12', '15', '18', '21'], size=10)
axes[3].plot(x_array, ubRMSE_array, color='green', linewidth=2)
axes[3].set_ylabel('ubRMSE')
axes[3].set_xlabel('Time of Day (UTC)', size=8)
plt.subplots_adjust(wspace=None, hspace=None)

plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp Statistics\n'+Plot_Labels_full, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_out_path+'ISMN5cm_All_Stations_StatisticsPlots'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)

##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 2 soil temps for each station 00Z
##################################################################################################################


figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))


filtered_x=ma.masked_outside(SCAN_zero_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_zero_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_zero_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_zero_array_soiltemp[:,:],mask=mask).compressed()
slope_00_L1, intercept_00_L1, r_value_00_L1, p_value_00_L1, std_err_00_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 00 UTC ISSCP vs. ISMN LY2 is...', r_value_00_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_00_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_00_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_00_L1)
ubrmse_00_L1=np.sqrt(rmse_00_L1**2-temp_bias_00_L1**2)

axes[0,0].set_ylim(Scat_min, Scat_max)
axes[0,0].set_xlim(Scat_min, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,0].plot(filtered_x_final, filtered_x_final*slope_00_L1+intercept_00_L1, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_00_L1,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_00_L1,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_00_L1,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_00_L1,2)), ha="right", va="center", size=10)

##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 2 soil temps for each station 03Z
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_three_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_three_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_three_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_three_array_soiltemp[:,:],mask=mask).compressed()
slope_03_L1, intercept_03_L1, r_value_03_L1, p_value_03_L1, std_err_03_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 03 UTC ISSCP vs. ISMN LY2 is...', r_value_03_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_03_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_03_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_03_L1)
ubrmse_03_L1=np.sqrt(rmse_03_L1**2-temp_bias_03_L1**2)

axes[1,0].set_ylim(240, Scat_max)
axes[1,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,0].plot(filtered_x_final, filtered_x_final*slope_03_L1+intercept_03_L1, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_03_L1,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_03_L1,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_03_L1,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_03_L1,2)), ha="right", va="center", size=10)

##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 2 soil temps for each station 06Z
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_six_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_six_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_six_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_six_array_soiltemp[:,:],mask=mask).compressed()
slope_06_L1, intercept_06_L1, r_value_06_L1, p_value_06_L1, std_err_06_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 06 UTC ISSCP vs. ISMN LY2 is...', r_value_06_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_06_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_06_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_06_L1)
ubrmse_06_L1=np.sqrt(rmse_06_L1**2-temp_bias_06_L1**2)

axes[2,0].set_ylim(240, Scat_max)
axes[2,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,0].plot(filtered_x_final, filtered_x_final*slope_06_L1+intercept_06_L1, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_06_L1,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_06_L1,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_06_L1,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_06_L1,2)), ha="right", va="center", size=10)

##################################################################################################################
#####  Generate an 8-panel scatter plot of ISCCP versus SCAN Layer 2 soil temps for each station 09Z
##################################################################################################################
filtered_x=ma.masked_outside(SCAN_nine_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_nine_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_nine_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_nine_array_soiltemp[:,:],mask=mask).compressed()
slope_09_L1, intercept_09_L1, r_value_09_L1, p_value_09_L1, std_err_09_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 09 UTC ISSCP vs. ISMN LY2 is...', r_value_09_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_09_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_09_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_09_L1)
ubrmse_09_L1=np.sqrt(rmse_09_L1**2-temp_bias_09_L1**2)

axes[3,0].set_ylim(240, Scat_max)
axes[3,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,0].plot(filtered_x_final, filtered_x_final*slope_09_L1+intercept_09_L1, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5,  "RMS :"+str(round(rmse_09_L1,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_09_L1,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_09_L1,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_09_L1,2)), ha="right", va="center", size=10)
    
##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 00Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twelve_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_twelve_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twelve_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twelve_array_soiltemp[:,:],mask=mask).compressed()
slope_12_L1, intercept_12_L1, r_value_12_L1, p_value_12_L1, std_err_12_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 12 UTC ISSCP vs. ISMN LY2 is...', r_value_12_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_12_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_12_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_12_L1)
ubrmse_12_L1=np.sqrt(rmse_12_L1**2-temp_bias_12_L1**2)

axes[0,1].set_ylim(240, Scat_max)
axes[0,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,1].plot(filtered_x_final, filtered_x_final*slope_12_L1+intercept_12_L1, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_12_L1,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_12_L1,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_12_L1,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_12_L1,2)), ha="left", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 06Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_fifteen_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_fifteen_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_fifteen_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_fifteen_array_soiltemp[:,:],mask=mask).compressed()
slope_15_L1, intercept_15_L1, r_value_15_L1, p_value_15_L1, std_err_15_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 15 UTC ISSCP vs. ISMN LY1 is...', r_value_15_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_15_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_15_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #= math.sqrt(mse_15_L1)
ubrmse_15_L1=np.sqrt(rmse_15_L1**2-temp_bias_15_L1**2)

axes[1,1].set_ylim(240, Scat_max)
axes[1,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,1].plot(filtered_x_final, filtered_x_final*slope_15_L1+intercept_15_L1, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_15_L1,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_15_L1,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_15_L1,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_15_L1,2)), ha="left", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 12Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_eighteen_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_eighteen_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_eighteen_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_eighteen_array_soiltemp[:,:],mask=mask).compressed()
slope_18_L1, intercept_18_L1, r_value_18_L1, p_value_18_L1, std_err_18_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 18 UTC ISSCP vs. ISMN LY2 is...', r_value_18_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_18_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_18_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.)) # = math.sqrt(mse_18_L1)
ubrmse_18_L1=np.sqrt(rmse_18_L1**2-temp_bias_18_L1**2)

axes[2,1].set_ylim(240, Scat_max)
axes[2,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,1].plot(filtered_x_final, filtered_x_final*slope_18_L1+intercept_18_L1, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_18_L1,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_18_L1,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_18_L1,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_18_L1,2)), ha="left", va="center", size=10)

##################################################################################################################
#####  PLOT THE Whole Dataset as a line plot for 18Z Top Layer COMPARISON
##################################################################################################################

filtered_x=ma.masked_outside(SCAN_twtyone_array_soiltemp[1,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(ISSCP_twtyone_array_soiltemp[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(SCAN_twtyone_array_soiltemp[1,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(ISSCP_twtyone_array_soiltemp[:,:],mask=mask).compressed()
slope_21_L1, intercept_21_L1, r_value_21_L1, p_value_21_L1, std_err_21_L1 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the 21 UTC ISSCP vs. ISMN LY2 is...', r_value_21_L1)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_21_L1=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_21_L1 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_21_L1=np.sqrt(rmse_21_L1**2-temp_bias_21_L1**2)

axes[3,1].set_ylim(240, Scat_max)
axes[3,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,1].plot(filtered_x_final, filtered_x_final*slope_21_L1+intercept_21_L1, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 10cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_21_L1,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_21_L1,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_21_L1,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_21_L1,2)), ha="left", va="center", size=10)
    
plt.suptitle('ISCCP Skin Temperature vs. ISMN 10cm Soil Temp\n'+Plot_Labels_full, fontsize=18)

#img_fname_pre=img_out_path+stationsfile['State Code'][station_number]+'-'+str(stationsfile['Stat_Name'][station_number])
img_fname_pre=img_out_path
plt.savefig(img_out_path+'ISCCPvsISMN10cm_All_Stations_'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)


##################################################################################################################
#####  Create a plot of all the Bias, RMSE, and Correlations
##################################################################################################################

figure, axes=plt.subplots(nrows=4, ncols=1, figsize=(8,8))

Bias_array=np.array((temp_bias_00_L1, temp_bias_03_L1, temp_bias_06_L1, temp_bias_09_L1, temp_bias_12_L1, temp_bias_15_L1, temp_bias_18_L1, temp_bias_21_L1))
CORR_array=np.array((r_value_00_L1, r_value_03_L1, r_value_06_L1, r_value_09_L1, r_value_12_L1, r_value_15_L1, r_value_18_L1, r_value_21_L1))
RMSE_array=np.array((rmse_00_L1, rmse_03_L1, rmse_06_L1, rmse_09_L1, rmse_12_L1, rmse_15_L1, rmse_18_L1, rmse_21_L1))
ubRMSE_array=np.array((ubrmse_00_L1, ubrmse_03_L1, ubrmse_06_L1, ubrmse_09_L1, ubrmse_12_L1, ubrmse_15_L1, ubrmse_18_L1, ubrmse_21_L1))

x_array=np.array((1,2,3,4,5,6,7,8))

axes[0].set_ylim(-20, 20)
axes[0].set_xlim(0, 9)
axes[0].grid(which='major', axis='y')
axes[0].set_xticks([0,1,2,3,4,5,6,7,8])
axes[0].set_xticklabels([])
axes[0].plot(x_array, Bias_array, color='red', linewidth=2)
axes[0].set_ylabel('Bias')


axes[1].set_ylim(-1, 1)
axes[1].set_xlim(0, 9)
axes[1].grid(which='major', axis='y')
axes[1].set_xticks([0,1,2,3,4,5,6,7,8])
axes[1].set_xticklabels([])
axes[1].plot(x_array, CORR_array, color='blue', linewidth=2)
axes[1].set_ylabel('Correlation')

axes[2].set_ylim(-10, 10)
axes[2].set_xlim(0, 9)
axes[2].grid(which='major', axis='y')
axes[2].set_xticks([0,1,2,3,4,5,6,7,8])
axes[2].set_xticklabels([])
axes[2].plot(x_array, RMSE_array, color='green', linewidth=2)
axes[2].set_ylabel('RMSE')
plt.subplots_adjust(wspace=None, hspace=None)

axes[3].set_ylim(-10, 10)
axes[3].set_xlim(0, 9)
axes[3].grid(which='major', axis='y')
axes[3].set_xticks([0,1,2,3,4,5,6,7,8])
axes[3].set_xticklabels([' ', '00', '03', '06', '09', '12', '15', '18', '21'], size=10)
axes[3].plot(x_array, ubRMSE_array, color='black', linewidth=2)
axes[3].set_ylabel('ubRMSE')
axes[3].set_xlabel('Time of Day (UTC)', size=8)
plt.subplots_adjust(wspace=None, hspace=None)

plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp Statistics\n'+Plot_Labels_full, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_out_path+'ISMN10cm_All_Stations_StatisticsPlots'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)

########################################################################################################################################################################
####   Winter  Plots of ISCCP vs. ISMN
########################################################################################################################################################################
####   00Z Winter
########################################################################################################################################################################
figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))

filtered_x=ma.masked_outside(Winter_00_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_00_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_00_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_00_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_00_L0, intercept_WinScan_00_L0, r_value_WinScan_00_L0, p_value_WinScan_00_L0, std_err_WinScan_00_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY2 is...', r_value_WinScan_00_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_00_L0=np.sum(filtered_diff)/filtered_diff.shape[0]

rmse_WinISCCP_00_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_00_L0)
ubrmse_WinISCCP_00_L0=np.sqrt(rmse_WinISCCP_00_L0**2-temp_bias_WinISCCP_00_L0**2)

axes[0,0].set_ylim(240, Scat_max)
axes[0,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,0].plot(filtered_x_final, filtered_x_final*slope_WinScan_00_L0+intercept_WinScan_00_L0, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)')
axes[0,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_WinScan_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_00_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   03Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_03_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_03_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_03_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_03_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_03_L0, intercept_WinScan_03_L0, r_value_WinScan_03_L0, p_value_WinScan_03_L0, std_err_WinScan_03_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY1 is...', r_value_WinScan_03_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_03_L0=np.sum(filtered_diff)/filtered_diff.shape[0]

rmse_WinISCCP_03_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))   #math.sqrt(mse_WinISCCP_03_L0)
ubrmse_WinISCCP_03_L0=np.sqrt(rmse_WinISCCP_03_L0**2-temp_bias_WinISCCP_03_L0**2)

axes[1,0].set_ylim(240, Scat_max)
axes[1,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,0].plot(filtered_x_final, filtered_x_final*slope_WinScan_03_L0+intercept_WinScan_03_L0, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)')
axes[1,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_WinScan_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_03_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   06Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_06_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_06_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_06_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_06_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_06_L0, intercept_WinScan_06_L0, r_value_WinScan_06_L0, p_value_WinScan_06_L0, std_err_WinScan_06_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY1 is...', r_value_WinScan_06_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_06_L0=np.sum(filtered_diff)/filtered_diff.shape[0]

rmse_WinISCCP_06_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_06_L0)
ubrmse_WinISCCP_06_L0=np.sqrt(rmse_WinISCCP_06_L0**2-temp_bias_WinISCCP_06_L0**2)

axes[2,0].set_ylim(240, Scat_max)
axes[2,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,0].plot(filtered_x_final, filtered_x_final*slope_WinScan_06_L0+intercept_WinScan_06_L0, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)')
axes[2,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_WinScan_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_06_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   09Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_09_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_09_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_09_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_09_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_09_L0, intercept_WinScan_09_L0, r_value_WinScan_09_L0, p_value_WinScan_09_L0, std_err_WinScan_09_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY2 is...', r_value_WinScan_09_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_09_L0=np.sum(filtered_diff)/filtered_diff.shape[0]

rmse_WinISCCP_09_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_09_L0)
ubrmse_WinISCCP_09_L0=np.sqrt(rmse_WinISCCP_09_L0**2-temp_bias_WinISCCP_09_L0**2)

axes[3,0].set_ylim(240, Scat_max)
axes[3,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,0].plot(filtered_x_final, filtered_x_final*slope_WinScan_09_L0+intercept_WinScan_09_L0, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_WinScan_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_09_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   12Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_12_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_12_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_12_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_12_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_12_L0, intercept_WinScan_12_L0, r_value_WinScan_12_L0, p_value_WinScan_12_L0, std_err_WinScan_12_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY1 is...', r_value_WinScan_12_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_12_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_WinISCCP_12_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_12_L0)
ubrmse_WinISCCP_12_L0=np.sqrt(rmse_WinISCCP_12_L0**2-temp_bias_WinISCCP_12_L0**2)

axes[0,1].set_ylim(240, Scat_max)
axes[0,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,1].plot(filtered_x_final, filtered_x_final*slope_WinScan_12_L0+intercept_WinScan_12_L0, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)')
axes[0,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_WinScan_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_12_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   15Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_15_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_15_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_15_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_15_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_15_L0, intercept_WinScan_15_L0, r_value_WinScan_15_L0, p_value_WinScan_15_L0, std_err_WinScan_15_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY1 is...', r_value_WinScan_15_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_15_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_WinISCCP_15_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_15_L0)
ubrmse_WinISCCP_15_L0=np.sqrt(rmse_WinISCCP_15_L0**2-temp_bias_WinISCCP_15_L0**2)

axes[1,1].set_ylim(240, Scat_max)
axes[1,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,1].plot(filtered_x_final, filtered_x_final*slope_WinScan_15_L0+intercept_WinScan_15_L0, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)')
axes[1,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_WinScan_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_15_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   18Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_18_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_18_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_18_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_18_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_18_L0, intercept_WinScan_18_L0, r_value_WinScan_18_L0, p_value_WinScan_18_L0, std_err_WinScan_18_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY1 is...', r_value_WinScan_18_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_18_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_WinISCCP_18_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_18_L0)
ubrmse_WinISCCP_18_L0=np.sqrt(rmse_WinISCCP_18_L0**2-temp_bias_WinISCCP_18_L0**2)

axes[2,1].set_ylim(240, Scat_max)
axes[2,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
axes[2,1].plot(filtered_x_final, filtered_x_final*slope_WinScan_18_L0+intercept_WinScan_18_L0, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)')
axes[2,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_WinScan_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_18_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   21Z Winter
########################################################################################################################################################################

filtered_x=ma.masked_outside(Winter_21_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Winter_21_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Winter_21_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Winter_21_ISCCP[:,:],mask=mask).compressed()
slope_WinScan_21_L0, intercept_WinScan_21_L0, r_value_WinScan_21_L0, p_value_WinScan_21_L0, std_err_WinScan_21_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Winter ISSCP vs. ISMN LY1 is...', r_value_WinScan_21_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_WinISCCP_21_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_WinISCCP_21_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_WinISCCP_21_L0)
ubrmse_WinISCCP_21_L0=np.sqrt(rmse_WinISCCP_21_L0**2-temp_bias_WinISCCP_21_L0**2)

axes[3,1].set_ylim(240, Scat_max)
axes[3,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,1].plot(filtered_x_final, filtered_x_final*slope_WinScan_21_L0+intercept_WinScan_21_L0, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_WinISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_WinISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_WinScan_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_WinISCCP_21_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp Winter (DJF)\n'+Plot_Labels_full, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_out_path+'Seasonal_Winter_ISCCPvsISMN5cm_All_Stations'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)

########################################################################################################################################################################
####   Spring  Plots of ISCCP vs. ISMN
########################################################################################################################################################################
####   00Z Spring
########################################################################################################################################################################


figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))

filtered_x=ma.masked_outside(Spring_00_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_00_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_00_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_00_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_00_L0, intercept_SprScan_00_L0, r_value_SprScan_00_L0, p_value_SprScan_00_L0, std_err_SprScan_00_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_00_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_00_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_00_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SprISCCP_00_L0)
ubrmse_SprISCCP_00_L0=np.sqrt(rmse_SprISCCP_00_L0**2-temp_bias_SprISCCP_00_L0**2)

axes[0,0].set_ylim(240, Scat_max)
axes[0,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,0].plot(filtered_x_final, filtered_x_final*slope_SprScan_00_L0+intercept_SprScan_00_L0, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)')
axes[0,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SprScan_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_00_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   03Z Spring
########################################################################################################################################################################

filtered_x=ma.masked_outside(Spring_03_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_03_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_03_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_03_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_03_L0, intercept_SprScan_03_L0, r_value_SprScan_03_L0, p_value_SprScan_03_L0, std_err_SprScan_03_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_03_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_03_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_03_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SprISCCP_03_L0)
ubrmse_SprISCCP_03_L0=np.sqrt(rmse_SprISCCP_03_L0**2-temp_bias_SprISCCP_03_L0**2)

axes[1,0].set_ylim(240, Scat_max)
axes[1,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,0].plot(filtered_x_final, filtered_x_final*slope_SprScan_03_L0+intercept_SprScan_03_L0, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)')
axes[1,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SprScan_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_03_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   06Z Spring
########################################################################################################################################################################

filtered_x=ma.masked_outside(Spring_06_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_06_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_06_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_06_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_06_L0, intercept_SprScan_06_L0, r_value_SprScan_06_L0, p_value_SprScan_06_L0, std_err_SprScan_06_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_06_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_06_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_06_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SprISCCP_06_L0)
ubrmse_SprISCCP_06_L0=np.sqrt(rmse_SprISCCP_06_L0**2-temp_bias_SprISCCP_06_L0**2)

axes[2,0].set_ylim(240, Scat_max)
axes[2,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,0].plot(filtered_x_final, filtered_x_final*slope_SprScan_06_L0+intercept_SprScan_06_L0, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)')
axes[2,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SprScan_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_06_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   09Z Spring
########################################################################################################################################################################

filtered_x=ma.masked_outside(Spring_09_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_09_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_09_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_09_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_09_L0, intercept_SprScan_09_L0, r_value_SprScan_09_L0, p_value_SprScan_09_L0, std_err_SprScan_09_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_09_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_09_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_09_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SprISCCP_09_L0)
ubrmse_SprISCCP_09_L0=np.sqrt(rmse_SprISCCP_09_L0**2-temp_bias_SprISCCP_09_L0**2)

axes[3,0].set_ylim(240, Scat_max)
axes[3,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,0].plot(filtered_x_final, filtered_x_final*slope_SprScan_09_L0+intercept_SprScan_09_L0, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SprScan_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_09_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   12Z Spring
########################################################################################################################################################################


filtered_x=ma.masked_outside(Spring_12_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_12_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_12_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_12_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_12_L0, intercept_SprScan_12_L0, r_value_SprScan_12_L0, p_value_SprScan_12_L0, std_err_SprScan_12_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_12_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_12_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_12_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SprISCCP_12_L0)
ubrmse_SprISCCP_12_L0=np.sqrt(rmse_SprISCCP_12_L0**2-temp_bias_SprISCCP_12_L0**2)

axes[0,1].set_ylim(240, Scat_max)
axes[0,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,1].plot(filtered_x_final, filtered_x_final*slope_SprScan_12_L0+intercept_SprScan_12_L0, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)')
axes[0,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SprScan_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_12_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   15Z Spring
########################################################################################################################################################################

filtered_x=ma.masked_outside(Spring_15_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_15_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_15_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_15_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_15_L0, intercept_SprScan_15_L0, r_value_SprScan_15_L0, p_value_SprScan_15_L0, std_err_SprScan_15_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_15_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_15_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_15_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SprISCCP_15_L0)
ubrmse_SprISCCP_15_L0=np.sqrt(rmse_SprISCCP_15_L0**2-temp_bias_SprISCCP_15_L0**2)

axes[1,1].set_ylim(240, Scat_max)
axes[1,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,1].plot(filtered_x_final, filtered_x_final*slope_SprScan_15_L0+intercept_SprScan_15_L0, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)')
axes[1,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SprScan_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_15_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   18Z Spring
########################################################################################################################################################################

filtered_x=ma.masked_outside(Spring_18_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_18_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_18_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_18_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_18_L0, intercept_SprScan_18_L0, r_value_SprScan_18_L0, p_value_SprScan_18_L0, std_err_SprScan_18_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY2 is...', r_value_SprScan_18_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_18_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_18_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_SprISCCP_18_L0)
ubrmse_SprISCCP_18_L0=np.sqrt(rmse_SprISCCP_18_L0**2-temp_bias_SprISCCP_18_L0**2)

axes[2,1].set_ylim(240, Scat_max)
axes[2,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,1].plot(filtered_x_final, filtered_x_final*slope_SprScan_18_L0+intercept_SprScan_18_L0, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)')
axes[2,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SprScan_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_18_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   21Z Spring
########################################################################################################################################################################

filtered_x=ma.masked_outside(Spring_21_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Spring_21_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Spring_21_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Spring_21_ISCCP[:,:],mask=mask).compressed()
slope_SprScan_21_L0, intercept_SprScan_21_L0, r_value_SprScan_21_L0, p_value_SprScan_21_L0, std_err_SprScan_21_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Spring ISSCP vs. ISMN LY1 is...', r_value_SprScan_21_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SprISCCP_21_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SprISCCP_21_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))   #math.sqrt(mse_SprISCCP_21_L0)
ubrmse_SprISCCP_21_L0=np.sqrt(rmse_SprISCCP_21_L0**2-temp_bias_SprISCCP_21_L0**2)

axes[3,1].set_ylim(240, Scat_max)
axes[3,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,1].plot(filtered_x_final, filtered_x_final*slope_SprScan_21_L0+intercept_SprScan_21_L0, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SprISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SprISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SprScan_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SprISCCP_21_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp Spring (MAM)\n'+Plot_Labels_full, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_out_path+'Seasonal_Spring_ISCCPvsISMN5cm_All_Stations'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)

########################################################################################################################################################################
####   Summer  Plots of ISCCP vs. ISMN
########################################################################################################################################################################
####   00Z Summer
########################################################################################################################################################################

figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))

filtered_x=ma.masked_outside(Summer_00_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_00_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_00_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_00_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_00_L0, intercept_SumScan_00_L0, r_value_SumScan_00_L0, p_value_SumScan_00_L0, std_err_SumScan_00_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_00_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_00_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
true_val = filtered_y_final
predicted_val = intercept_SumScan_00_L0 + slope_SumScan_00_L0 * filtered_x_final
mse_SumISCCP_00_L0 = mean_squared_error(true_val, predicted_val)
rmse_SumISCCP_00_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SumISCCP_00_L0)
ubrmse_SumISCCP_00_L0=np.sqrt(rmse_SumISCCP_00_L0**2-temp_bias_SumISCCP_00_L0**2)

axes[0,0].set_ylim(240, Scat_max)
axes[0,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,0].plot(filtered_x_final, filtered_x_final*slope_SumScan_00_L0+intercept_SumScan_00_L0, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)')
axes[0,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SumScan_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_00_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   03Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_03_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_03_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_03_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_03_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_03_L0, intercept_SumScan_03_L0, r_value_SumScan_03_L0, p_value_SumScan_03_L0, std_err_SumScan_03_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_03_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_03_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
true_val = filtered_y_final
predicted_val = intercept_SumScan_03_L0 + slope_SumScan_03_L0 * filtered_x_final
mse_SumISCCP_03_L0 = mean_squared_error(true_val, predicted_val)
rmse_SumISCCP_03_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SumISCCP_03_L0)
ubrmse_SumISCCP_03_L0=np.sqrt(rmse_SumISCCP_03_L0**2-temp_bias_SumISCCP_03_L0**2)

axes[1,0].set_ylim(240, Scat_max)
axes[1,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,0].plot(filtered_x_final, filtered_x_final*slope_SumScan_03_L0+intercept_SumScan_03_L0, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)')
axes[1,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SumScan_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_03_L0,3)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   06Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_06_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_06_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_06_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_06_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_06_L0, intercept_SumScan_06_L0, r_value_SumScan_06_L0, p_value_SumScan_06_L0, std_err_SumScan_06_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_06_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_06_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
true_val = filtered_y_final
predicted_val = intercept_SumScan_06_L0 + slope_SumScan_06_L0 * filtered_x_final
mse_SumISCCP_06_L0 = mean_squared_error(true_val, predicted_val)
rmse_SumISCCP_06_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  #math.sqrt(mse_SumISCCP_06_L0)
ubrmse_SumISCCP_06_L0=np.sqrt(rmse_SumISCCP_06_L0**2-temp_bias_SumISCCP_06_L0**2)

axes[2,0].set_ylim(240, Scat_max)
axes[2,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,0].plot(filtered_x_final, filtered_x_final*slope_SumScan_06_L0+intercept_SumScan_06_L0, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)')
axes[2,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SumScan_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_06_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   09Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_09_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_09_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_09_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_09_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_09_L0, intercept_SumScan_09_L0, r_value_SumScan_09_L0, p_value_SumScan_09_L0, std_err_SumScan_09_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_09_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_09_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
true_val = filtered_y_final
predicted_val = intercept_SumScan_09_L0 + slope_SumScan_09_L0 * filtered_x_final
mse_SumISCCP_09_L0 = mean_squared_error(true_val, predicted_val)
rmse_SumISCCP_09_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_SumISCCP_09_L0)
ubrmse_SumISCCP_09_L0=np.sqrt(rmse_SumISCCP_09_L0**2-temp_bias_SumISCCP_09_L0**2)

axes[3,0].set_ylim(240, Scat_max)
axes[3,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,0].plot(filtered_x_final, filtered_x_final*slope_SumScan_09_L0+intercept_SumScan_09_L0, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_SumScan_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_09_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   12Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_12_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_12_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_12_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_12_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_12_L0, intercept_SumScan_12_L0, r_value_SumScan_12_L0, p_value_SumScan_12_L0, std_err_SumScan_12_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_12_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_12_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
true_val = filtered_y_final
predicted_val = intercept_SumScan_12_L0 + slope_SumScan_12_L0 * filtered_x_final
mse_SumISCCP_12_L0 = mean_squared_error(true_val, predicted_val)
rmse_SumISCCP_12_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_SumISCCP_12_L0)
ubrmse_SumISCCP_12_L0=np.sqrt(rmse_SumISCCP_12_L0**2-temp_bias_SumISCCP_12_L0**2)

axes[0,1].set_ylim(240, Scat_max)
axes[0,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[0,1].plot(filtered_x_final, filtered_x_final*slope_SumScan_12_L0+intercept_SumScan_12_L0, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)')
axes[0,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SumScan_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_12_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   15Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_15_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_15_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_15_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_15_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_15_L0, intercept_SumScan_15_L0, r_value_SumScan_15_L0, p_value_SumScan_15_L0, std_err_SumScan_15_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_15_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_15_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
true_val = filtered_y_final
predicted_val = intercept_SumScan_15_L0 + slope_SumScan_15_L0 * filtered_x_final
mse_SumISCCP_15_L0 = mean_squared_error(true_val, predicted_val)
rmse_SumISCCP_15_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_SumISCCP_15_L0)
ubrmse_SumISCCP_15_L0=np.sqrt(rmse_SumISCCP_15_L0**2-temp_bias_SumISCCP_15_L0**2)

axes[1,1].set_ylim(240, Scat_max)
axes[1,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,1].plot(filtered_x_final, filtered_x_final*slope_SumScan_15_L0+intercept_SumScan_15_L0, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)')
axes[1,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SumScan_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_15_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   18Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_18_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_18_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_18_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_18_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_18_L0, intercept_SumScan_18_L0, r_value_SumScan_18_L0, p_value_SumScan_18_L0, std_err_SumScan_18_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY2 is...', r_value_SumScan_18_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_18_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SumISCCP_18_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_SumISCCP_18_L0=np.sqrt(rmse_SumISCCP_18_L0**2-temp_bias_SumISCCP_18_L0**2)

axes[2,1].set_ylim(240, Scat_max)
axes[2,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,1].plot(filtered_x_final, filtered_x_final*slope_SumScan_18_L0+intercept_SumScan_18_L0, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)')
axes[2,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SumScan_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_18_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   21Z Summer
########################################################################################################################################################################

filtered_x=ma.masked_outside(Summer_21_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Summer_21_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Summer_21_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Summer_21_ISCCP[:,:],mask=mask).compressed()
slope_SumScan_21_L0, intercept_SumScan_21_L0, r_value_SumScan_21_L0, p_value_SumScan_21_L0, std_err_SumScan_21_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Summer ISSCP vs. ISMN LY1 is...', r_value_SumScan_21_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_SumISCCP_21_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_SumISCCP_21_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_SumISCCP_21_L0=np.sqrt(rmse_SumISCCP_21_L0**2-temp_bias_SumISCCP_21_L0**2)

axes[3,1].set_ylim(240, Scat_max)
axes[3,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,1].plot(filtered_x_final, filtered_x_final*slope_SumScan_21_L0+intercept_SumScan_21_L0, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_SumISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_SumISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_SumScan_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_SumISCCP_21_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp Summer (JJA)\n'+Plot_Labels_full, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_out_path+'Seasonal_Summer_ISCCPvsISMN5cm_All_Stations'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)

########################################################################################################################################################################
####   Fall  Plots of ISCCP vs. ISMN
########################################################################################################################################################################
####   00Z Fall
########################################################################################################################################################################

figure, axes=plt.subplots(nrows=4, ncols=2, figsize=(16,8))

filtered_x=ma.masked_outside(Fall_00_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_00_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_00_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_00_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_00_L0, intercept_FallScan_00_L0, r_value_FallScan_00_L0, p_value_FallScan_00_L0, std_err_FallScan_00_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_00_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_00_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_00_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_FallISCCP_00_L0=np.sqrt(rmse_FallISCCP_00_L0**2-temp_bias_FallISCCP_00_L0**2)

axes[0,0].set_ylim(240, Scat_max)
axes[0,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
axes[0,0].plot(filtered_x_final, filtered_x_final*slope_FallScan_00_L0+intercept_FallScan_00_L0, color='red', linewidth=2)
axes[0,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,0].grid()
axes[0,0].set_ylabel('ISCCP Skin T (K)')
axes[0,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,0].text(Scat_min+2,  Scat_max-10, "00 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[0,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_FallScan_00_L0,2)), ha="right", va="center", size=10)
axes[0,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_00_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   03Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_03_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_03_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_03_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_03_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_03_L0, intercept_FallScan_03_L0, r_value_FallScan_03_L0, p_value_FallScan_03_L0, std_err_FallScan_03_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_03_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_03_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_03_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_FallISCCP_03_L0=np.sqrt(rmse_FallISCCP_03_L0**2-temp_bias_FallISCCP_03_L0**2)

axes[1,0].set_ylim(240, Scat_max)
axes[1,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
axes[1,0].plot(filtered_x_final, filtered_x_final*slope_FallScan_03_L0+intercept_FallScan_03_L0, color='red', linewidth=2)
axes[1,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,0].grid()
axes[1,0].set_ylabel('ISCCP Skin T (K)')
axes[1,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,0].text(Scat_min+2,  Scat_max-10, "03 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[1,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_FallScan_03_L0,2)), ha="right", va="center", size=10)
axes[1,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_03_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   06Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_06_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_06_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_06_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_06_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_06_L0, intercept_FallScan_06_L0, r_value_FallScan_06_L0, p_value_FallScan_06_L0, std_err_FallScan_06_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_06_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_06_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_06_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_FallISCCP_06_L0)
ubrmse_FallISCCP_06_L0=np.sqrt(rmse_FallISCCP_06_L0**2-temp_bias_FallISCCP_06_L0**2)

axes[2,0].set_ylim(240, Scat_max)
axes[2,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
axes[2,0].plot(filtered_x_final, filtered_x_final*slope_FallScan_06_L0+intercept_FallScan_06_L0, color='red', linewidth=2)
axes[2,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,0].grid()
axes[2,0].set_ylabel('ISCCP Skin T (K)')
axes[2,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,0].text(Scat_min+2,  Scat_max-10, "06 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[2,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_FallScan_06_L0,2)), ha="right", va="center", size=10)
axes[2,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_06_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   09Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_09_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_09_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_09_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_09_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_09_L0, intercept_FallScan_09_L0, r_value_FallScan_09_L0, p_value_FallScan_09_L0, std_err_FallScan_09_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_09_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_09_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_09_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_FallISCCP_09_L0=np.sqrt(rmse_FallISCCP_09_L0**2-temp_bias_FallISCCP_09_L0**2)

axes[3,0].set_ylim(240, Scat_max)
axes[3,0].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,0].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
axes[3,0].plot(filtered_x_final, filtered_x_final*slope_FallScan_09_L0+intercept_FallScan_09_L0, color='red', linewidth=2)
axes[3,0].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,0].grid()
axes[3,0].set_ylabel('ISCCP Skin T (K)')
axes[3,0].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,0].text(Scat_min+2,  Scat_max-10, "09 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,0].text(Scat_min-12, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-22, "(ISSCP-ISMN)", ha="right", va="center", size=6)
axes[3,0].text(Scat_min-12, Scat_max-35, "R2:"+str(round(r_value_FallScan_09_L0,2)), ha="right", va="center", size=10)
axes[3,0].text(Scat_min-12, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_09_L0,2)), ha="right", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,0].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   12Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_12_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_12_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_12_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_12_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_12_L0, intercept_FallScan_12_L0, r_value_FallScan_12_L0, p_value_FallScan_12_L0, std_err_FallScan_12_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_12_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_12_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_12_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))   # math.sqrt(mse_FallISCCP_12_L0)
ubrmse_FallISCCP_12_L0=np.sqrt(rmse_FallISCCP_12_L0**2-temp_bias_FallISCCP_12_L0**2)

axes[0,1].set_ylim(240, Scat_max)
axes[0,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')
axes[0,1].plot(filtered_x_final, filtered_x_final*slope_FallScan_12_L0+intercept_FallScan_12_L0, color='red', linewidth=2)
axes[0,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[0,1].grid()
axes[0,1].set_ylabel('ISCCP Skin T (K)')
axes[0,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[0,1].text(Scat_min+2, Scat_max-10, "12 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[0,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[0,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_FallScan_12_L0,2)), ha="left", va="center", size=10)
axes[0,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_12_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[0,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   15Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_15_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_15_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_15_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_15_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_15_L0, intercept_FallScan_15_L0, r_value_FallScan_15_L0, p_value_FallScan_15_L0, std_err_FallScan_15_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_15_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_15_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
#compute the RMSD
rmse_FallISCCP_15_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_FallISCCP_15_L0)
#compute the ubRMSD
ubrmse_FallISCCP_15_L0=np.sqrt(rmse_FallISCCP_15_L0**2-temp_bias_FallISCCP_15_L0**2)

axes[1,1].set_ylim(240, Scat_max)
axes[1,1].set_xlim(240, Scat_max)
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[1,1].plot(filtered_x_final, filtered_x_final*slope_FallScan_15_L0+intercept_FallScan_15_L0, color='red', linewidth=2)
axes[1,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[1,1].grid()
axes[1,1].set_ylabel('ISCCP Skin T (K)')
axes[1,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[1,1].text(Scat_min+2, Scat_max-10, "15 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[1,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[1,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_FallScan_15_L0,2)), ha="left", va="center", size=10)
axes[1,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_15_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[1,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   18Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_18_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_18_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_18_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_18_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_18_L0, intercept_FallScan_18_L0, r_value_FallScan_18_L0, p_value_FallScan_18_L0, std_err_FallScan_18_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_18_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_18_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_18_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))  # math.sqrt(mse_FallISCCP_18_L0)
ubrmse_FallISCCP_18_L0=np.sqrt(rmse_FallISCCP_18_L0**2-temp_bias_FallISCCP_18_L0**2)

axes[2,1].set_ylim(240, Scat_max)
axes[2,1].set_xlim(240, Scat_max)

xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[2,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[2,1].plot(filtered_x_final, filtered_x_final*slope_FallScan_18_L0+intercept_FallScan_18_L0, color='red', linewidth=2)
axes[2,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[2,1].grid()
axes[2,1].set_ylabel('ISCCP Skin T (K)')
axes[2,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[2,1].text(Scat_min+2, Scat_max-10, "18 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[2,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[2,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_FallScan_18_L0,2)), ha="left", va="center", size=10)
axes[2,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_18_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[2,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)

########################################################################################################################################################################
####   21Z Fall
########################################################################################################################################################################

filtered_x=ma.masked_outside(Fall_21_SCAN[0,:,:],min_cor,max_cor)
filtered_y=ma.masked_outside(Fall_21_ISCCP[:,:],min_cor,max_cor)
mask=ma.masked_invalid(filtered_x.filled(np.nan)*filtered_y.filled(np.nan)).mask
filtered_x_final=ma.masked_array(Fall_21_SCAN[0,:,:],mask=mask).compressed()
filtered_y_final=ma.masked_array(Fall_21_ISCCP[:,:],mask=mask).compressed()
slope_FallScan_21_L0, intercept_FallScan_21_L0, r_value_FallScan_21_L0, p_value_FallScan_21_L0, std_err_FallScan_21_L0 = stats.linregress(filtered_x_final, filtered_y_final)
print ('this r value for the Fall ISSCP vs. ISMN LY1 is...', r_value_FallScan_21_L0)

#compute the bias
filtered_diff=filtered_y_final-filtered_x_final
temp_bias_FallISCCP_21_L0=np.sum(filtered_diff)/filtered_diff.shape[0]
rmse_FallISCCP_21_L0 = np.sqrt(np.nanmean((filtered_y_final-filtered_x_final)**2.))
ubrmse_FallISCCP_21_L0=np.sqrt(rmse_FallISCCP_21_L0**2-temp_bias_FallISCCP_21_L0**2)
std_dev=np.std(filtered_diff)

axes[3,1].set_ylim(240, Scat_max)
axes[3,1].set_xlim(240, Scat_max)
#axes[3,1].scatter(ISSCP_twtyone_array_soiltemp[:,:], SCAN_twtyone_array_soiltemp[1,:,:], marker='+')
#axes[3,1].scatter(filtered_x_final, filtered_y_final, marker='+')
xy = np.vstack([filtered_x_final, filtered_y_final])
z = gaussian_kde(xy)(xy)
axes[3,1].scatter(filtered_x_final, filtered_y_final, c=z, s=10, marker='+', cmap='turbo')

axes[3,1].plot(filtered_x_final, filtered_x_final*slope_FallScan_21_L0+intercept_FallScan_21_L0, color='red', linewidth=2)
axes[3,1].plot([0,Scat_max],[0, Scat_max], color='black', linewidth=2)
axes[3,1].grid()
axes[3,1].set_ylabel('ISCCP Skin T (K)')
axes[3,1].set_xlabel('ISMN 5cm Tsoil (K)', size=8)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
axes[3,1].text(Scat_min+2, Scat_max-10, "21 UTC", ha="left", va="center", size=12, bbox=bbox_props)
axes[3,1].text(Scat_max+3, Scat_max-5, "RMS :"+str(round(rmse_FallISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-15, "BIAS:"+str(round(temp_bias_FallISCCP_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-22, "(ISSCP-ISMN)", ha="left", va="center", size=6)
axes[3,1].text(Scat_max+3, Scat_max-35, "R2:"+str(round(r_value_FallScan_21_L0,2)), ha="left", va="center", size=10)
axes[3,1].text(Scat_max+3, Scat_max-45, "ubRMSD:"+str(round(ubrmse_FallISCCP_21_L0,2)), ha="left", va="center", size=10)
bbox_props = dict(boxstyle="square", fc="w", ec="0.5", alpha=0.9)
pts=str(filtered_diff.shape[0])
axes[3,1].text(Scat_max-2, Scat_min+10, 'number of points='+pts, ha="right", va="center", size=10, bbox=bbox_props)
    
plt.suptitle('ISCCP Skin Temperature vs. ISMN 5cm Soil Temp Fall (SON)\n'+Plot_Labels_full, fontsize=18)
img_fname_pre=img_out_path
plt.savefig(img_out_path+'Seasonal_Fall_ISCCPvsISMN5cm_All_Stations'+BGDATE+'-'+EDATE+'_'+Plot_Labels+'.png')
plt.close(figure)
