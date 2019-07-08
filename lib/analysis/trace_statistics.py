# ----------------------------------------------------------------
# IoT Netprofiler
# Licensed under The MIT License [see LICENSE for details]
# Written by Luca Maiano - https://www.linkedin.com/in/lucamaiano/
# ----------------------------------------------------------------


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def node_adftest(packets_node, topology, experiment):
    # Compute Augmented Dickey-Fuller test
    adftest = pd.DataFrame(columns=['topology', 'experiment', 'node', 'adf statistic', 'p-value',
      'critical values 1%', 'critical-values 5%', 'critical-values 10%', 'stationary'])

    for node in packets_node.keys():
        X = packets_node[node]['rtt'].dropna().values
        try:
          result = adfuller(X)
        except:
          continue
        adf = result[0]
        p_value = result[1]
        critical_value_1 = result[4]['1%']
        critical_value_5 = result[4]['5%']
        critical_value_10 = result[4]['10%']
        
        if not adftest.empty:
            df = pd.DataFrame({
              'topology': [topology],
              'experiment': [experiment],
              'node': [node],
              'adf statistic': [adf], 
              'p-value': [p_value], 
              'critical values 1%': [critical_value_1], 
              'critical-values 5%': [critical_value_5], 
              'critical-values 10%': [critical_value_10],
              'stationary': [p_value <= 0.05] })
            adftest = adftest.append(df)
        else:
            adftest = pd.DataFrame(data={
              'topology': [topology],
              'experiment': [experiment],
              'node': [node],
              'adf statistic': [adf], 
              'p-value': [p_value], 
              'critical values 1%': [critical_value_1], 
              'critical-values 5%': [critical_value_5], 
              'critical-values 10%': [critical_value_10],
              'stationary': [p_value <= 0.05] })

    return adftest
