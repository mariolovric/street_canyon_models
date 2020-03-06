from src import *


def process():
    df = pd.read_csv('../data/preprocessed.csv', index_col=0)
    #set time index
    df.index = pd.DatetimeIndex(df.index)
    df = df.rename(columns=names_dict)

    # select used variables from list sel2
    X0 = df[['Temperature max', 'Temperature min', 'Temperature average', 'Temperature max-min',
              'Pressure max',  'Pressure average',
              'Relative humidity max', 'Relative humidity min', 'Relative humidity average',
              'Relative humidity max-min',
              'Wind speed', 'Precipitation', 'Wind direction',
              'EC ug/m3', 'OC ug/m3', 'PM10 ug/m3', 'NO2 ug/m3',
              'Month', 'Year', 'Season', 'Weekday']]
    # add day before /lag

    lag_list1 = ['EC ug/m3', 'OC ug/m3', 'PM10 ug/m3', 'NO2 ug/m3', 'Weekday','Temperature max', 'Temperature min', 'Temperature average',
                 'Temperature max-min', 'Pressure max', 'Pressure average', 'Relative humidity max', 'Relative humidity min', 'Relative humidity average',
                 'Relative humidity max-min', 'Wind speed', 'Precipitation', 'Wind direction']

    #create lag values
    X1 = X0[lag_list1].shift(+1)
    X1.columns = [str(col) + '_lag' for col in X1.columns]
    #concat lag to X
    X = pd.concat([X0, X1], sort=True, axis=1)
    X.dropna(how='all', inplace=True)

    return X

#dictionary for mapping long and short names
names_dict={'meteo_temp_max':'Temperature max', 'meteo_temp_min':'Temperature min', 'meteo_temp_mean':'Temperature average',
       'meteo_temp_max-min':'Temperature max-min', 'meteo_pressure_max':'Pressure max', 'meteo_pressure_min':'Pressure min',
       'meteo_pressure_mean':'Pressure average', 'meteo_pressure_max-min':'Pressure max-min', 'meteo_relhum_max':'Relative humidity max',
       'meteo_relhum_min':'Relative humidity min', 'meteo_relhum_mean':'Relative humidity average', 'meteo_relhum_max-min':'Relative humidity max-min',
       'meteo_wind_speed':'Wind speed', 'meteo_precipitation':'Precipitation', 'prtcl_elem_carbon_inPM10':'EC/PM10',
       'prtcl_org_carbon_inPM10':'OC/PM10', 'prtcl_tot_carbon_inPM10':'TOC/PM10','prtcl_prim_carbon_unit':'POC',
       'prtcl_sec_org_carbon_inOC':'SOC/OC', 'prtcl_elem_carbon_unit':'EC ug/m3',
       'prtcl_org_carbon_unit':'OC ug/m3', 'prtcl_org_by_elem_carbon':'OC/EC', 'prtcl_pm10_unit':'PM10 ug/m3',
       'prtcl_sec_carbon_unit':'SOC ug/m3', 'prtcl_tot_carbon_unit':'TOC ug/m3', 'no2':'NO2 ug/m3', 'weekdays':'Weekday',
       'months':'Month', 'year':'Year', 'season':'Season', 'meteo_wind_direction_numerical':'Wind direction'}



