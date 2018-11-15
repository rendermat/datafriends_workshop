import numpy as np
import scipy.stats as sc
import pandas as pd

def add_sums_to_dataframe(df_raw):
    '''
    Add the horizontal and vertical sums to a dataframe which holds just the observed values.
    '''
    df_raw['h_sum'] = df_raw.sum(axis=1)
    df_v_sum = pd.DataFrame(columns=['v_sum'], data=df_raw.sum(axis=0)).transpose()
    df_raw = df_raw.append(df_v_sum)
    return df_raw


def calculate_expected_dataframe(df_obs):
    '''
    Calculates the expected values to a dataframe with observed values.
    The dateframe of observed values has to include vertical and horizontal sums.
    '''
    index = df_obs.index
    columns = df_obs.columns
    
    # Copy the observed values dataframe as a template.
    df_exp = df_obs.copy()
    # This is needed in case the observed values are integers. We will get floats for the expected ones.
    for column in columns:
        df_exp[column] = pd.to_numeric(df_exp[column], downcast='float')
    
    # Sort out the sum row and column labels for easier access.
    v_sum_row = index[len(index) - 1]
    h_sum_column = columns[len(columns) - 1]
    total_sum = df_obs.at[v_sum_row,h_sum_column]

    # Now calculate the expected values and set them to the expected values dataframe.
    for column in columns[:-1]:
        for row in index[:-1]:
            exp_value = float(df_obs.at[row,h_sum_column] * df_obs.at[v_sum_row,column] / total_sum)
            df_exp.at[row,column] = exp_value

    return df_exp


def chi_squared(df_obs, df_exp):
    '''
    Calculated chi squared for a dataframe of observed values and a dataframe of expected values.
    '''
    chi_squared = 0
    
    for row in df_obs.index[:2]:
        for column in df_obs.columns[:5]:
            h_obs = df_obs.at[row, column]
            h_exp = df_exp.at[row, column]
            h_diff = h_obs - h_exp
            chi_squared = chi_squared + h_diff**2 / h_exp
            
    return chi_squared 