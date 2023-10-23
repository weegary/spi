# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:38:35 2023

@author: Gary
"""

import numpy as np
from scipy import stats as st

"""
#  Standardized Index for SPI, SSI, SGI, etc.
"""

class Standardized_Index:
    def __init__(self, data,time):
        # 滾動式調整移動平均
        data_ma = data.rolling(time,center=False).mean() # rolling 計算時間窗口

        # 移動平均取natural log
        data_log = np.log(data_ma + 1e-5) # 1e-5 為改變浮點數精度, 解決0的問題
        data_log[ np.isinf(data_log) == True] = np.nan  # 更改值為NaN, 因為log0無意義, 會出現error
        
        # 全部的移動平均的平均, 忽略nan
        mu_data_ma = np.nanmean((data_ma))
        
        # 取log後的移動平均總合
        sum_log = np.nansum(data_log)

        # 計算 Gamma distrubution 參數
        n = len(data_log[time-1:])                     # 資料大小
        A = np.log(mu_data_ma + 1e-5) - (sum_log/n)    # 計算 A
        alpha = (1/(4*A))*(1+(1+((4*A)/3))**0.5)       # 計算 alpha (a)
        beta = mu_data_ma/alpha                        # 計算 beta (scale) 
        
        # 根據 Gamma distrubution 求取累積機率函數(Cumulative distribution function, CDF)
        gamma = st.gamma.cdf(data_ma + 1e-5, a=alpha, scale=beta)
        
        # 再轉換為標準常態分布 (Inverse of CDF)
        # Percent point function (inverse of cdf — percentiles).
        norm_spi = st.norm.ppf(gamma, loc=0, scale=1)  # ppf是將CDF轉換的語法, loc是平均值, scale是標準差
        
        self.moving_averaged_data = data_ma
        self.natural_log_of_moving_averaged_data = data_log
        self.mean_of_moving_averaged_data = mu_data_ma
        self.sum_of_natural_log = sum_log
        self.count_of_moving_averaged_data = n
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.gamma_distribution = gamma
        self.spi = norm_spi
