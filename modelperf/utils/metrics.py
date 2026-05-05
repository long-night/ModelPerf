"""Metrics calculation utilities for validation."""

from typing import Union, List, Optional, Any
import math


def calculate_mape(predicted: float, actual: float) -> float:
    """Mean Absolute Percentage Error.
    
    MAPE = (|predicted - actual| / |actual|) * 100
    
    Returns percentage (0-100).
    """
    if actual == 0:
        return float('inf') if predicted != 0 else 0.0
    return abs((predicted - actual) / actual) * 100


def calculate_rmse(predicted, actual):
    """Root Mean Square Error.
    
    RMSE = sqrt(mean((predicted - actual)^2))
    """
    if isinstance(predicted, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(predicted) != len(actual):
            raise ValueError("Lists must have same length")
        squared_errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]
        mse = sum(squared_errors) / len(squared_errors)
    else:
        mse = (predicted - actual) ** 2
    
    return math.sqrt(mse)


def calculate_mae(predicted, actual):
    """Mean Absolute Error.
    
    MAE = mean(|predicted - actual|)
    """
    if isinstance(predicted, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(predicted) != len(actual):
            raise ValueError("Lists must have same length")
        absolute_errors = [abs(p - a) for p, a in zip(predicted, actual)]
        return sum(absolute_errors) / len(absolute_errors)
    else:
        return abs(predicted - actual)


def calculate_r2(predicted, actual):
    """R-squared (Coefficient of Determination).
    
    R^2 = 1 - (SS_res / SS_tot)
    
    where SS_res = sum((actual - predicted)^2)
          SS_tot = sum((actual - mean(actual))^2)
    
    Returns value between 0 and 1 (1 is perfect fit).
    """
    if len(predicted) != len(actual):
        raise ValueError("Lists must have same length")
    
    n = len(actual)
    if n < 2:
        return 1.0 if predicted == actual else 0.0
    
    mean_actual = sum(actual) / n
    
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_error_summary(predicted: float, actual: float) -> dict:
    """Calculate all error metrics for a single value comparison.
    
    Returns dict with mape, rmse, mae.
    """
    return {
        'mape': calculate_mape(predicted, actual),
        'rmse': calculate_rmse(predicted, actual),
        'mae': calculate_mae(predicted, actual),
    }


__all__ = [
    'calculate_mape',
    'calculate_rmse',
    'calculate_mae',
    'calculate_r2',
    'calculate_error_summary',
]