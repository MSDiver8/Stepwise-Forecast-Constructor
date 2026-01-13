"""Модуль для запуска моделей и формирования конструктора пошагового автоматического прогноза."""

import warnings
from math import floor
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import statsmodels.formula.api as smf

class PsevdoForecastError(Exception):
    """Ошибка, возникающая при посторение псевдо прогноза моделями"""
class RealForecastError(Exception):
    """Ошибка, возникающая при посторение реального прогноза моделями"""
class AutoParamsSelectionError(Exception):
    """Ошибка, возникающая при автоматическом выборе параметров для базовых функций прогнозирования"""
class AutoRealForecastError(Exception):
    """Ошибка, возникающая при посторение автоматического реального прогноза по моделям"""
class PsevdoForecastTestMAPError(Exception):
    """Ошибка, возникающая в псевдовневыборочном тесте"""

# Модели для псевдовневыборочного прогноза
def ps_RW(Data: dict, Deep_forecast_period: int, Forecast_horizon: int):
    '''
    Функция строит псевдовневыборочный наивный прогноз.
    Наивный прогноз (модель случайного блуждания, RW). Прогноз в момент времени T - H на h = 1, ..., n шагов,
    n - горизонт прогнозирования,
    будет равен значению показателя в этот момент времени и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df.loc[:, 'obs']) - Deep_forecast_period  # базовый период, на котором оценивается модель
    quantity_pseudo_forecasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
    forecast_table = [] # двумерный массив прогнозов
    for i in range(quantity_pseudo_forecasts):
        forecast_table.append([df.iloc[base_period - 1 + i, 1] for _ in range(Forecast_horizon)]) # из дф берется последняя известная точка, прогноз - значения этой точки, итог массив длиной forecast_horizon с одним и тем же числом.
    ## групировка по шагам
    steps_table=[] # таблица прогнозов по шагам. каждому столбцу соответсвует номер шага
    for i in range(Forecast_horizon):
        steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_forecasts)])
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)} # записываем в словарь
    df = pd.DataFrame.from_dict(dic)
    return(df)

def ps_RWS(Data: dict, Deep_forecast_period: int, Forecast_horizon: int, Seasonality : int):
    '''
    Функция строит псевдовневыборочный наивный сезонный прогноз.
    Наивный сезонный прогноз (модель сезонного случайного блуждания, RWS). Прогноз в момент времени T - H на h = 1, ...,  n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени минус коэфицент сезонности и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df.loc[:, 'obs']) - Deep_forecast_period # длина базового периода
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов  
    forecast_table = [] # двумерный массив прогнозов
    for i in range(quantity_pseudo_foracasts):
        forecast_table.append(df.iloc[base_period - Seasonality + i : base_period - Seasonality + i + Forecast_horizon, 1].to_list()) # из дф берется масив с данными смещенными на один месяц, так для каждого момента прогнозирования
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):
        steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    return(df)

def ps_RWD(Data: dict, Deep_forecast_period: int, Forecast_horizon: int, Window_in_years: int = None ):
    '''
    Функция строит псевдовневыборочный наивный прогноз с дрейфом.
    Наивный прогноз c дрейфом (модель случайного блуждания с дрейфом, RWD). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
    df.index = df.loc[:, 'date'] # Индекс дата
    df = df.drop('date', axis = 1)
    base_period =  len(df.loc[:, 'obs']) - Deep_forecast_period
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
    # составление двумерного массива ркурсивного прогноза 
    if Window_in_years == None or Window_in_years == 0:
        forecast_table = []
        const_RWD_r = df.loc[:, 'obs'].diff().mean()
        for i in range(quantity_pseudo_foracasts):
            forecast_table.append([round(df.iloc[base_period - 1 + i, 0] + (j + 1) * const_RWD_r, 2) for j in range(Forecast_horizon)]) # формула модели RWD, заполняется масив для каждого момента прогнозирования. Затем это записывается в общий масив прогнозов.
    # скользящее окно       
    else:
        if Window_in_years < 2:
            raise PsevdoForecastError('Окно слишком мало')
        window = 12 * Window_in_years
        forecast_table = []
        for i in range(quantity_pseudo_foracasts):
            const_RWD_w = df.loc[:, 'obs'][base_period - window + i:base_period + i].diff().mean()
            forecast_table.append([round(df.iloc[base_period - 1 + i, 0] + (j + 1) * const_RWD_w, 2) for j in range(Forecast_horizon)])
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):
        steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    return(round(df,2))

def ps_RWDS(Data: dict, Deep_forecast_period: int, Forecast_horizon: int, Seasonality : int = 0, Window_in_years: int = None):
    '''
    Функция строит псевдовневыборочный наивный сезонный прогноз с дрейфом.
    Наивный сезонный прогноз c дрейфом (модель случайного сезонного блуждания с дрейфом, RWDS). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени мину сезоннный сдвиг и плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df.loc[:, 'obs']) - Deep_forecast_period # длина базового периода
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
    forecast_list = [] # список прогнозов
    # составление двумерного массива ркурсивного прогноза 
    if Window_in_years == None or Window_in_years == 0:
        for i in range(quantity_pseudo_foracasts):
            forecast_list.append(
                [
                round(
                    df.iloc[base_period + j - Seasonality - 1 + i, 1] +
                    df.iloc[:base_period + j, 1].diff(Seasonality).mean(),
                    2)
                for j in range(Forecast_horizon)
                ]
            )
    # скользящее окно       
    else:                                                                                                          
        if Window_in_years < 2:
            raise PsevdoForecastError('Окно слишком мало')
        else:
            window = 12 * Window_in_years
            for i in range(quantity_pseudo_foracasts):
                forecast_list.append(
                    [
                    round(
                        df.iloc[base_period + j - Seasonality + 1 + i, 1] +
                        df.iloc[base_period + j - window:base_period + j, 1].diff(Seasonality).mean(),
                        2
                        )
                    for j in range(Forecast_horizon)
                    ]
                )
    # групировка по шагам
    steps_list=[]
    for i in range(Forecast_horizon):
        steps_list.append([forecast_list[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_list[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    return df

def ps_TS(Data: dict, Deep_forecast_period: int, Forecast_horizon: int, Window_in_years: int = None, Reassessment: bool = False ):
    '''
    Функция строит псевдовневыборочный прогноз используя модель линейного тренда TS.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    df['T'] = [i + 1 for i in range(len(df.loc[:, 'obs']))]
    df.obs = df.obs.astype(float)
    base_period =  len(df) - Deep_forecast_period
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
    forecast_list = []
    if Window_in_years == None or Window_in_years == 0:
        if Reassessment == True:
            for i in range(quantity_pseudo_foracasts):
                alpha = smf.ols('obs ~ T', data=df.iloc[:base_period + i,[1, 2]]).fit().params['Intercept'] # рассчитываем константу 1
                betta = smf.ols('obs ~ T', data=df.iloc[:base_period + i,[1, 2]]).fit().params['T'] # рассчитываем константу 2
                forecast_list.append([round(alpha + df.iloc[base_period + i + j, 2] * betta, 2) for j in range(Forecast_horizon)])
        else:
            alpha = smf.ols('obs ~ T', data=df.iloc[:base_period,[1, 2]]).fit().params['Intercept'] # рассчитываем константу 1
            betta = smf.ols('obs ~ T', data=df.iloc[:base_period,[1, 2]]).fit().params['T'] # рассчитываем константу 2
            for i in range(quantity_pseudo_foracasts):
                forecast_list.append([round(alpha + df.iloc[base_period + i + j, 2] * betta, 2) for j in range(Forecast_horizon)])
    else:
        if Window_in_years < 2:
            raise PsevdoForecastError('Окно слишком мало')
        window = 12 * Window_in_years
        for i in range(quantity_pseudo_foracasts):
            alpha = smf.ols('obs ~ T', data=df.iloc[base_period - window + i:base_period + i,[1, 2]]).fit().params['Intercept'] # рассчитываем константу 1
            betta = smf.ols('obs ~ T', data=df.iloc[base_period - window + i:base_period + i,[1, 2]]).fit().params['T'] # рассчитываем константу 2
            forecast_list.append([round(alpha + df.iloc[base_period + i + j, 2] * betta, 2) for j in range(Forecast_horizon)])
    # групировка по шагам
    steps_list=[]
    for i in range(Forecast_horizon):
        steps_list.append([forecast_list[j][i] for j in range(quantity_pseudo_foracasts)])
    dic = { f's{i+1}' : steps_list[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    df = round(df,2)
    return(df)

# Модели для реального прогноза
def RW_real(Data: dict, Forecast_horizon: int):
    '''
    Функция строит реальный наивный прогноз.
    Наивный прогноз (модель случайного блуждания, RW). Прогноз в момент времени T   H на h = 1, ..., 12 шагов,
    будет равен значению показателя в этот момент времени и т.д.
    Возвращает:
    pandas.Series — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    RW_forecast_list = [df.iloc[-1,1] for _ in range(Forecast_horizon)] # из дф берется последняя известная точка, прогноз - значения этой точки, итог массив длиной forecast_horizon с одним и тем же числом.
    # Создание временной даты соответсвующей прогнозным значениям
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq)
    results = pd.Series(data = RW_forecast_list, index = Date, name = 'predicted')
    return(results)

def RWS_real(Data: dict, Forecast_horizon: int, Seasonality: int):
    '''
    Функция строит реальный наивный сезонный прогноз.
    Наивный сезонный прогноз (модель сезонного случайного блуждания, RWS). Прогноз в момент времени T - H на h = 1, ...,  n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени минус коэфицент сезонности и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    if Seasonality < Forecast_horizon:
        raise RealForecastError('Seasonality < Forecast_horizon, Функция подходит только для горизонта прогноза не привышающего величину сезонного сдвига')
    df = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    RWS_forecast_list = [df.iloc[- Seasonality + i ,1] for i in range(Forecast_horizon)] # из дф берется масив с данными смещенными на один месяц, так для каждого момента прогнозирования
    # Создание временной даты соответсвующей прогнозным значениям
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq )
    results = pd.Series(data = RWS_forecast_list, index = Date, name = 'predicted')
    return(results)

def RWD_real(Data: dict, Forecast_horizon: int, Window_in_years: int = None):
    '''
    Функция строит реальный наивный прогноз с дрейфом.
    Наивный прогноз c дрейфом (модель случайного блуждания с дрейфом, RWD). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    df.obs = df.obs.astype(float) # значения переводятся в формат float                                                               
    # рекурсивный
    if Window_in_years == None or Window_in_years == 0:
        const_RWD_r = df.loc[:, 'obs'].diff().mean()
        RWD_forecast_list = [round(df.iloc[-1, 1] + (j + 1) * const_RWD_r, 2) for j in range(Forecast_horizon)] # формула модели RWD
        # Создание временной даты соответсвующей прогнозным значениям
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq )
        results = pd.Series(data = RWD_forecast_list, index = Date, name = 'predicted')
        return(results)
    # скользящее окно       
    else:                                                                                                          
        if Window_in_years < 2:
            return print('слишком маленькое окно')
        window = 12 * Window_in_years
        const_RWD_w = df.loc[:, 'obs'][ - window:].diff().mean() # оценка константы происходит только на последних известных значениях попадающих в окно
        RWD_forecast_list = [round(df.iloc[-1, 1] + (j + 1) * const_RWD_w, 2) for j in range(Forecast_horizon)] 
        # Создание временной даты соответсвующей прогнозным значениям
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq )
        results = pd.Series(data = RWD_forecast_list, index = Date, name = 'predicted')
        return(results)
    
def RWDS_real(Data: dict, Forecast_horizon: int, Seasonality: int = 0, Window_in_years: int = None):
    '''
    Функция строит реальныйнаивный сезонный прогноз с дрейфом.
    Наивный сезонный прогноз c дрейфом (модель случайного сезонного блуждания с дрейфом, RWDS). Прогноз в момент времени T - H на h = 1, ..., n шагов, n - горизонт прогнозирования, 
    будет равен значению показателя в этот момент времени мину сезоннный сдвиг и плюс коэфицент "дрейфа" и т.д.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    if Seasonality < Forecast_horizon:
        raise RealForecastError('Seasonality < Forecast_horizon, Функция подходит только для горизонта прогноза не привышающего величину сезонного сдвига')
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    df.obs = df.obs.astype(float)   
    if Window_in_years == None or Window_in_years == 0:                                                           
        RWDS_forecast_list = [round(df.iloc[- Seasonality - 1 + i, 1] + df.obs.diff(Seasonality).mean(), 2) for i in range(Forecast_horizon)]
        # Создание временной даты соответсвующей прогнозным значениям
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq)
        results = pd.Series(data = RWDS_forecast_list, index = Date, name = 'predicted')
        return(results)
    # скользящее окно
    else:
        if Window_in_years < 2:
            raise RealForecastError('Окно слишеом мало, следует ставить более 2 лет')
        else:
            window = 12 * Window_in_years
            RWDS_forecast_list = [round(df.iloc[- Seasonality - 1 + i, 1] + df.obs[-window:].diff(Seasonality).mean(), 2) for i in range(Forecast_horizon)]
         # Создание временной даты соответсвующей прогнозным значениям
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq)
        results = pd.Series(data = RWDS_forecast_list, index = Date, name = 'predicted')
        return(results)
    
def TS_real(Data: dict, Forecast_horizon: int, Window_in_years: int = None):
    '''
    Функция строит реальный прогноз используя модель линейного тренда TS.
    Возвращает:
    pandas.DatsFrame — прогноз длиной Forecast_horizon с датами в индексе.
    '''
    df: pd.DataFrame = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    df['T'] = [i + 1 for i in range(len(df.loc[:, 'obs']))]
    df.obs = df.obs.astype(float)
    if Window_in_years == None or Window_in_years == 0:
        alpha = smf.ols('obs ~ T', data=df.iloc[:,[1, 2]]).fit().params['Intercept'] # рассчитываем константу 1
        betta = smf.ols('obs ~ T', data=df.iloc[:,[1, 2]]).fit().params['T'] # рассчитываем константу 2
        TS_forecast_list = [round(alpha + (df[-1:].index.to_list()[0] + i) * betta, 2) for i in range(Forecast_horizon)]
        # Создание временной даты соответсвующей прогнозным значениям
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq)
        results = pd.Series(data = TS_forecast_list, index = Date, name = 'predicted')
        return(results)
    else:
        if Window_in_years < 2:
            raise RealForecastError('Окно слишком мало')
        window = 12 * Window_in_years
        alpha = smf.ols('obs ~ T', data=df.iloc[ - window:,[1, 2]]).fit().params['Intercept'] # рассчитываем константу 1
        betta = smf.ols('obs ~ T', data=df.iloc[ - window:,[1, 2]]).fit().params['T'] # рассчитываем константу 2
        TS_forecast_list = [round(alpha + (df[-1:].index.to_list()[0] + i) * betta, 2) for i in range(Forecast_horizon)]
        # Создание временной даты соответсвующей прогнозным значениям
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
        Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]), periods = Forecast_horizon, freq = Freq)
        results = pd.Series(data = TS_forecast_list, index = Date, name = 'predicted')
        return(results)
    
# Фукция автоматического подбора параметров для базовых моделей
def Auto_params_selection(Data: dict):
    
    """
    Автоматически подбирает базовые параметры для моделей прогнозирования по данным ряда.

    Функция:
    1) определяет частоту временного ряда (месячная, квартальная, годовая);
    2) рассчитывает:
       - Forecast_horizon — горизонт псевдовневыборочного прогноза,
       - Deep_forecast_period — длину псевдовневыборочного периода,
       - Seasonality — сезонность (при наличии),
       - Window_size — ширину скользящего окна;
    3) использует фиксированные экспертные значения и долю от длины ряда для коротких временных рядов;
    4) проверяет корректность частоты и при невозможности подобрать параметр возбуждает Auto_params_selection_Error.

    Возвращает:
        dict — словарь с ключами 'Forecast_horizon', 'Deep_forecast_period',
               'Seasonality', 'Window_size'.
    """
    
    df = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    
    dic_auto_params = {'Forecast_horizon': [], # Горизонт псевдовневыборочного прогноза
                       'Deep_forecast_period':[], # Длина псевдовневыборочного прогноза
                       'Seasonality': [], # параметр сезонности, учавствующий в уравнении модели (дополнительный снос)
                       'Window_size': []} # ширина скользящего окна
    # Экспертами устанавливаются фиксированные значения для Forecast horizon
    #Для коротких рядов допускается взятие 20% от длины ряда
    horizon_values = {'horizon_m': 12, 'horizon_q': 4, 'horizon_y': 3}
    ratio_horizon_to_len = floor(0.2 * len(df)) # Отношение горизонта псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if Freq == 'ME':
        if ratio_horizon_to_len > 12:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_m']   # ряд стандартной длины
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len     # для коротких рядов
    elif Freq in ('Q', 'QE'):
        if ratio_horizon_to_len > 4:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_q']
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len
    elif Freq == 'YE':
        if ratio_horizon_to_len > 3:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_y']
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len
    else:
        raise AutoParamsSelectionError("Невозможно определить горизонт псевдовневыборочного прогноза")
    
    # Экспертами устанавливаются фиксированные значения для Deep forecast period
    # Для коротких рядов допускается взятие 40% от длины ряда
    
    deep_forecast_values = {'deep_forecast_period_m': 24,
                            'deep_forecast_period_q': 8,
                            'deep_forecast_period_y': 6}
    # Для ME пороговое значение 60 точек. для 61 - прогнозный период 24, => 61-24=37 - период оценки.
    # Если период оценки должен быть не менее 24 точек (для сезонных моделей) => ряд длиной не менее 40 точек. 
    ratio_deep_forcast_period_to_len = floor(0.4 * len(df)) # Отношение длины псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if Freq == 'ME':    
        if ratio_deep_forcast_period_to_len > 24:   # изменено с 12 , так что бы 24 было 40% от ддлины ряда Для других частотнсотей так же
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_m']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    elif Freq in ('Q', 'QE'):
        if ratio_deep_forcast_period_to_len > 8:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_q']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    elif Freq == 'YE':
        if ratio_deep_forcast_period_to_len > 6:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_y']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    else:
        raise AutoParamsSelectionError("Невозможно определить длину псевдовневыборочного прогноза")
    
    # Экспертами устанавливаются фиксированные значения для Seasonality
    seasonality_values = {'seasonality_m': 12,  # для годовых рядов отсутствует сезонности
                          'seasonality_q': 4,
                          'seasonality_y' : None}
    if Freq == 'ME':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_m']
    elif Freq in ('Q', 'QE'):
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_q']
    elif Freq == 'YE':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_y']

    # Экспертами устанавливаются фиксированные значения для Windowsize
    windowsize_values = {'windowsize_m': 36,
                         'windowsize_q': 12,
                         'windowsize_y': 3}
    ratio_windowsize_to_len = floor(0.2 * len(df)) # Отношение ширины скользящего окна псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if Freq == 'ME':
        if ratio_windowsize_to_len > 12:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_m']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    elif Freq in ('Q', 'QE'):
        if ratio_windowsize_to_len > 4:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_q']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    elif Freq == 'YE':
        if ratio_windowsize_to_len > 3:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_y']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    else:
        raise AutoParamsSelectionError("Невозможно определить ширину скользящего окна")
    return dic_auto_params

# Функция подсчета усредненных  MAPE по шагам
def MAPE_step_by_step(Data: dict,
                      Dataframe_model: pd.DataFrame,
                      Deep_forecast_period: int,
                      Forecast_horizon: int):
    """
    Вычисляет усреднённые пошаговые значения MAPE для прогнозной модели.
    Функция:
    1) формирует реальные и модельные значения для каждого шага псевдовневыборочного периода;
    2) рассчитывает MAPE отдельно для каждого шага горизонта;
    3) возвращает список ошибок, где i-й элемент — MAPE на i-м шаге.
    Возвращает:
        list — значения MAPE длиной Forecast_horizon.
    """
    df_real = pd.DataFrame(Data['observations'])
    df_real.obs = df_real.obs.astype(float) # значения переводятся в формат float
    df_model = Dataframe_model.copy()
    train_period =  len(df_real.obs) - Deep_forecast_period
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1
    Real_Data_list = [df_real.obs[train_period+i:train_period+quantity_pseudo_foracasts+i] for i in range(Forecast_horizon)]
    #print(Real_Data_list[0])
    Model_Data_list = [df_model.iloc[:,i] for i in range(Forecast_horizon)]
    #print(Model_Data_list[0])
    Errors = []
    for i in range(Forecast_horizon):
        Errors.append(round(mean_absolute_percentage_error(Real_Data_list[i], Model_Data_list[i]), 2))
    return Errors

# Функция для формирования набора моделей в зависимости от длины и частотности ряда.
def select_models_for_series(Data: dict):
    """
    Функция для формирования набора моделей в зависимости от длины и частотности ряда.
    На вход: словарь с данными о ряде
    На выход: список моделей для которых будет производится псевдовневыборочный тест. 
    """
    
    df = pd.DataFrame(Data['observations'])
    if len(df) < 5:
        raise PsevdoForecastTestMAPError("Длина ряда меньше 5 значений")
        
    # Таблица соответствия 
    rules = {
        "ME": [
            (5,  ["RW", "RWD"]),
            (6,  ["RW", "RWD", "TS"]),
            (12, ["RW", "RWD", "TS"]),
            (24, ["RW", "RWD", "TS", "RWS", "RWDS"]),
            (25, ["RW", "RWD", "TS", "RWS", "RWDS"]),
            (27, ["RW", "RWD", "TS", "RWS", "RWDS"])
        ],
        "QE": [
            (5,  ["RW", "RWD"]),
            (6,  ["RW", "RWD", "TS"]),
            (12, ["RW", "RWD", "TS", "RWS", "RWDS"]),
            (24, ["RW", "RWD", "TS", "RWS", "RWDS"]),
            (25, ["RW", "RWD", "TS", "RWS", "RWDS"])
        ],
        "YE": [
            (5,  ["RW", "RWD"]),
            (6,  ["RW", "RWD", "TS"]),
            (12, ["RW", "RWD", "TS"]),
            (24, ["RW", "RWD", "TS"])
        ],
    }
    #results = []
    n = len(pd.DataFrame(Data['observations'])['obs'])
    freq = Data['frequency']
    # нормализация частоты
    if freq == "Q":
        freq = "QE"
    if freq not in ("ME", "QE", "YE"):
        raise PsevdoForecastTestMAPError("Неизвестная частотность")
    
    if freq not in rules:
        raise PsevdoForecastTestMAPError(f"Неизвестная частотность: {freq}")
    # Находим минимальный порог, который <= n
    available_models = []
    for threshold, models in rules[freq]:
        if n >= threshold:
            available_models = models  # обновляем (берем наибольший подходящий)
    #results.append({
    #    "pandas_frequency": freq,
    #    "observations_count": n,
    #    "available_models": available_models
    #})
    return available_models

# Функция определения минимального MAPE на каждом шаге и формирование списка с "лучшими" моделями для каждого шага прогнозирования
def Psevdo_forecast_test_MAPE(Data: dict,
                              Deep_forecast_period: int,
                              Forecast_horizon: int,
                              Seasonality: int,
                              Reassessment: bool):
    """
    Определяет модель с минимальным MAPE на каждом шаге псевдовневыборочного прогноза.
    Функция:
    1) строит псевдопрогнозы только для тех моделей, которые доступны в зависимости от длины и частоты ряда;
    2) вычисляет пошаговый MAPE для каждой модели;
    3) на каждом шаге горизонта выбирает модель с минимальной ошибкой;
    4) формирует и возвращает список моделей, оптимальных для каждого шага.
    Возвращает:
        list — последовательность имён моделей длиной Forecast_horizon.
    """
    Window_in_years = None # задел на будущее
    available_models = select_models_for_series(Data) # определяет список моделей доступный в зависимости от длины ряда
    # словарь “кандидатов”, где каждая модель — лямбда-функция
    candidate_models = {
        'RW': lambda: ps_RW(Data, Deep_forecast_period, Forecast_horizon),
        'RWS': lambda: ps_RWS(Data, Deep_forecast_period, Forecast_horizon, Seasonality),
        'RWD': lambda: ps_RWD(Data, Deep_forecast_period, Forecast_horizon, Window_in_years),
        'RWDS': lambda: ps_RWDS(Data, Deep_forecast_period, Forecast_horizon, Seasonality, Window_in_years),
        'TS': lambda: ps_TS(Data, Deep_forecast_period, Forecast_horizon, Window_in_years, Reassessment)
    }
    Forecast_dict = {}
    for name in available_models:
        if name not in candidate_models:
            warnings.warn(f"Модель '{name}' отсутствует в candidate_models и будет пропущена.")
            continue
        try:
            Forecast_dict[name] = candidate_models[name]()
        except Exception as e: # pylint: disable=broad-except
            warnings.warn(f"Модель '{name}' упала и будет пропущена: {e}")
    # После цикла:
    Error_dict = {k: MAPE_step_by_step(Data, Forecast_dict[k], Deep_forecast_period, Forecast_horizon) for k in Forecast_dict}
    List_of_model_number = [min(Error_dict, key=lambda key: Error_dict[key][i]) for i in range(Forecast_horizon)] # функция min() примененная к словарю обращается к его ключам. С помощью lambda функции выбирается сначала только первые элементы, затем только вторые и тд. затем функция min() выбирает минимальный элемент среди первых, затем среди вторых и тд, когда выбирается миниму, выражение возвращает ключ в котором он содержится
    return List_of_model_number
    
# Функция Автоматического прогноза - Конструктор моделей
def Auto_forecast(Data : dict,
                  Deep_research : bool=False):
    """
    Автоматически строит комбинированный прогноз временного ряда.
    Функция:
    1) преобразует входные данные и определяет частоту ряда;
    2) автоматически подбирает параметры прогноза (горизонт, сезонность и др.);
    3) выполняет псевдовневыборочный тест (MAPE) и выбирает лучшие модели;
    4) пошагово формирует прогноз, используя оптимальную модель для каждого шага;
    5) собирает итоговый результат (даты, шаги, модели, значения) в DataFrame.
    Параметр Deep_research позволяет переоценивать модель на каждом шаге псевдоневыборочного прогноза.
    Возвращает:
        pandas.DataFrame — прогноз по шагам с указанием используемых моделей.
    """
    df = pd.DataFrame(Data['observations'])
    Freq = Data['frequency']
    # Создается 2 словоря available_models и model_args, это делается для оптимизации работы функции.
    # Для построени комбинированного по шагам прогноза будут задействованы только необходимые функции,
    # которые получаются в результате работы функции Psevdo_forecast_test_MAPE. 
    # Функции для построения реального прогноза
    available_models = {
        'RW': RW_real,
        'RWS': RWS_real,
        'RWD': RWD_real,
        'RWDS': RWDS_real,
        'TS': TS_real
    }
    # Параметры функций для построения реального прогноза
    model_args = {
        'RW': ['Data', 'Forecast_horizon'],
        'RWS': ['Data', 'Forecast_horizon', 'Seasonality'],
        'RWD': ['Data', 'Forecast_horizon'],
        'RWDS': ['Data', 'Forecast_horizon', 'Seasonality'],
        'TS': ['Data', 'Forecast_horizon']
    }
    # dic_auto_params - Словарь с автоматически продобранными значениями базовых прогнозных функций. 
    dic_auto_params = Auto_params_selection(Data)
    # Для ускорения будем выключать переоценку моделей при построение псевдовневыборочного прогноза
    #if Deep_research == True:
    #    Reassessment = False
    #else:
    #    Reassessment = True
    #print(Reassessment)
    # List_of_model_number - Список с результатами по псевдовневыборочному тесту 
    # то есть, список моделей, которые участвуют в построение реального прогноза.
    List_of_model_number = Psevdo_forecast_test_MAPE(Data = Data,
                                                     Deep_forecast_period = dic_auto_params['Deep_forecast_period'],
                                                     Forecast_horizon = dic_auto_params['Forecast_horizon'],
                                                     Seasonality = dic_auto_params['Seasonality'], 
                                                     Reassessment = Deep_research) 
    # Конструирование реального прогноза по шагам прогнозирования. 
    Forecast, Model_name, Steps = [], [], []    # Для записи результатов 
    param_pool = {'Data': Data, **dic_auto_params}
    for i, model_name in enumerate(List_of_model_number):
        model_func = available_models[model_name]
        required_args = model_args[model_name]
        kwargs = {arg: param_pool[arg] for arg in required_args}
        result = model_func(**kwargs)
        idx = min(i, len(result) - 1)  # «прилипает» к последнему элементу
        Forecast.append(result.iloc[idx].round(2))
        Model_name.append(List_of_model_number[i])
        Steps.append(f'Горизонт {i+1}')
    # Создание временной даты соответсвующей прогнозным значениям
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(df.iloc[-1,0] + (df.iloc[-1,0] - df.iloc[-2,0]),
                         periods = dic_auto_params['Forecast_horizon'],
                         freq = Freq
                         ).strftime('%d.%m.%Y').tolist()
    # Записываем итоговый результат в виде датафрейма
    results = pd.DataFrame({
    'Дата': Date,
    'Шаги': Steps,
    'Модель': Model_name,
    'Прогноз': Forecast
    })
    return results