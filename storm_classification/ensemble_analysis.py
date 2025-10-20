#Ensemble analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.colors import Normalize
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, brier_score_loss
import os
from storm_utils.config_paths import get_project_paths
from datetime import timedelta


def evaluate_predictions(y_pred, y_pred_persistence, y_pred_27_day_recurrence, y_true, threshold=0.5):
    """
    Evaluate predictions using various classification metrics.

    Parameters:
    - y_pred: array, predicted probabilities
    - y_true: array, true labels
    - y_pred_persistence: array, persistence model predicted probabilities
    - threshold: float, threshold for classification (default: 0.5)

    Returns:
    - metrics_dict: dictionary, containing classification metrics
    """
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten().astype(int)
    y_pred_persistence = np.array(y_pred_persistence).flatten()
    
    if y_pred.shape != y_true.shape or y_pred_persistence.shape != y_true.shape:
        raise ValueError("Shape mismatch between predictions, persistence predictions, and true labels.")

    y_pred_labels = (y_pred > threshold).astype(int)

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    # ROC AUC
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred_labels).ravel()

    # Skill scores
    peirce_ss = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    pearson_ss = (TP + TN) / (TP + FP + TN + FN)
    heidke_ss = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

    # Brier Scores
    bs_forecast = brier_score_loss(y_true, y_pred)
    bs_climatology = brier_score_loss(y_true, np.full_like(y_true, np.mean(y_true)))
    bs_persistence = brier_score_loss(y_true, y_pred_persistence)
    bs_twentyseven = brier_score_loss(y_true, y_pred_27_day_recurrence)

    # Brier Skill Scores
    bss_climatology = 1 - (bs_forecast / bs_climatology) if bs_climatology > 0 else np.nan
    bss_persistence = 1 - (bs_forecast / bs_persistence) if bs_persistence > 0 else np.nan
    bss_twentyseven = 1 - (bs_forecast / bs_twentyseven) if bs_twentyseven > 0 else np.nan

    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Peirce Skill Score': peirce_ss,
        'Pearson Skill Score': pearson_ss,
        'Heidke Skill Score': heidke_ss,
        'Brier Score': bs_forecast,
        'BS Climatology': bs_climatology,
        'Brier Skill Score (Climatology)': bss_climatology,
        'BS Persistence': bs_persistence,
        'Brier Skill Score (Persistence)': bss_persistence,
        'BS 27 Day Recurrence': bs_twentyseven,
        'Brier Skill Score (27 Day Recurrence)': bss_twentyseven,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
    }

    return metrics_dict


def plot_window_data(window, min_offset, max_offset, forecast=None, persistence=None, twentyseven=None):
    fontsize=40
    # Get one sample from the datase
    v = window['v']                     # T0 - 24h -> T0 + 48h
    omni = window['omni_plotting']      # T0 - 24h -> T0 + 48h
    target = window['target_plotting']  # T0 - 24h -> T0 + 48h
    max_target = window['max_target']   # Target variable
    T0 = window['T0']
    F_start = window['target_start']
    F_end = window['target_end']

    correct = None
    if forecast is not None:
        correct = int(forecast + 0.5) == int(max_target > 4.66)
    
    # Create time axis
    start_time = T0 + timedelta(minutes=30 * min_offset)
    end_time = T0 + timedelta(minutes=30 * max_offset)

    timestamps = pd.date_range(start=start_time, periods=max_offset - min_offset, freq='30min')

    # Plot
    fig, axs = plt.subplots(2, figsize=(12, 8), sharex=True)
    ax1, ax2 = axs
    for i in range(v.shape[1]):
        label = 'v_ensemble' if i == 0 else None
        ax1.plot(timestamps, v[:, i], 'blue', lw=1, alpha=0.1, label=label)
    
    ax1.plot(timestamps, omni, 'k', lw=1.5, label='OMNI')
    ax2.plot(timestamps, target, '#E66100', lw=1.5, label='Hp30')

    ax1.vlines(T0, min((min(v.flatten()), min(omni))) - 50, max((max(v.flatten()), max(omni))) + 50, colors='black', linestyles='dashed', label='$T_0$')
    ax2.vlines(T0, 0, max(max(target)+1, 6), colors='black', linestyles='dashed', label='$T_0$')
    
        # --- Find maximum Hp30 within forecast window ---
    # Slice timestamps & target together into a DataFrame
    target_df = pd.DataFrame({"time": timestamps, "target": target})

    window_df = target_df[(target_df["time"] >= F_start) & (target_df["time"] <= F_end)]

    if not window_df.empty:
        idx_max = window_df["target"].idxmax()
        max_time = window_df.loc[idx_max, "time"]
        max_value = window_df.loc[idx_max, "target"]

        # Plot the point
        ax2.scatter(max_time, max_value, color="royalblue", s=80, zorder=5,
                    label=f"max Hp30={max_value:.2f}")


    ax2.hlines(4.66, timestamps[0], timestamps[-1], colors='#D41159', label='Storm Threshold=4.66', linestyles='dashed')

    GREEN = (0.6, 0.9, 0.6, 0.3)
    RED = (0.9, 0.6, 0.6, 0.3)
    bg_colour = 'white'
    if correct is not None:
        bg_colour =  GREEN if correct else RED
    # Format x-axis to show year

    for ax in axs:
        num_days = (timestamps[-1] - timestamps[0]).days
        interval = max(1, num_days // 6)  # roughly 6 ticks total
    
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.axvspan(F_start, F_end, color='grey', alpha=0.5)
        ax.set_facecolor(bg_colour)
        ax.tick_params(axis='x', labelsize=fontsize-5)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.grid(True)
    
        for label in ax.get_xticklabels():
            label.set_rotation(15)

    ax1.set_ylim(min((min(v.flatten()), min(omni))) - 50, max((max(v.flatten()), max(omni))) + 50)
    ax2.set_ylim(0, max(max(target)+1, 6))

    ax2.set_xlabel("Time (UTC)", fontsize=fontsize)
    ax1.set_ylabel('v (km/s)', fontsize=fontsize)
    ax2.set_ylabel('Hp30', fontsize=fontsize)

    persistence = float(persistence)
    twentyseven = float(twentyseven)
    
    if forecast is not None:
        ax1.set_title(f"Weighted Mean={forecast:.3f}, Persistence={persistence}, 27-day Recurrence={twentyseven}{' '*15}", fontsize=fontsize-15, pad=20)


def plot_model_seen_data(window, min_offset, max_offset, forecast=None, persistence=None, twentyseven=None):
    # Get one sample from the datase
    v = window['v']                     # T0 - 24h -> T0 + 48h
    omni = window['omni_plotting']      # T0 - 24h -> T0 + 48h
    target = window['target_plotting']  # T0 - 24h -> T0 + 48h
    max_target = window['max_target']   # Target variable
    T0 = window['T0']
    F_start = window['target_start']
    F_end = window['target_end']

    correct = None
    if forecast is not None:
        correct = int(forecast + 0.5) == int(max_target > 4.66)
    
    # Create time axis
    start_time = T0 + timedelta(minutes=30 * min_offset)
    end_time = T0 + timedelta(minutes=30 * max_offset)
    timestamps = pd.date_range(start=start_time, periods=max_offset - min_offset, freq='30min')
    # Plot
    fig, axs = plt.subplots(2, figsize=(10, 6))
    ax1, ax2 = axs
    for i in range(v.shape[1]):
        label = 'v_ensemble' if i == 0 else None
        ax1.plot(timestamps, v[:, i], 'blue', lw=1, alpha=0.1, label=label)
    
    ax1.plot(timestamps[:48], omni[:48], 'k', lw=1, label='OMNI')
    ax2.plot(timestamps[:48], target[:48], lw=1, label='Hp30')

    ax1.vlines(T0, min((min(v.flatten()), min(omni))) - 50, max((max(v.flatten()), max(omni))) + 50, colors='black', linestyles='dashed', label='T0')
    ax2.vlines(T0, 0, max(target) + 1, colors='black', linestyles='dashed', label='T0')
    # ax2.hlines(max_target, F_start, F_end, colors='red', label=f'max Hp30 ={max_target:.2f}')

    ax2.hlines(4.66, timestamps[0], timestamps[-1], colors='dodgerblue', label='Storm Threshold=4.66', linestyles='dotted')

    GREEN = (0.6, 0.9, 0.6, 0.5)
    RED = (0.9, 0.6, 0.6, 0.5)
    bg_colour = 'white'
    if correct is not None:
        bg_colour =  GREEN if correct else RED
    # Format x-axis to show year
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.set_xlabel("Time", weight='bold')
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.axvspan(F_start, F_end, color='grey', alpha=0.3)
        ax.set_facecolor(bg_colour)

    ax1.set_ylim(min((min(v.flatten()), min(omni))) - 50, max((max(v.flatten()), max(omni))) + 50)
    ax2.set_ylim(0, max(target, fontweight='bold')+1)

    ax1.set_ylabel('v (km/s)', weight='bold')
    ax2.set_ylabel('Hp30', weight='bold')
    
    # Optional: auto-rotate for readability
    fig.autofmt_xdate()

    if forecast is not None:
        ax1.set_title(f"Ensemble v Forecast Around T0 = {T0}\nModel={forecast:.3f}:{correct}, Persistence={persistence}, 27-day Recurrence={twentyseven}", fontsize=18)
    
    plt.show()


def persistence_plot(window, min_offset, max_offset):
    v = window['v']                     # T0 - 24h -> T0 + 48h
    target = window['target_plotting']  # T0 - 24h -> T0 + 48h
    max_target = window['max_target']   # Target variable
    T0 = window['T0']
    F_start = window['target_start']
    F_end = window['target_end']

    persistence = target[min_offset * 2]
    print(persistence)

    # Create time axis
    start_time = T0 + timedelta(minutes=30 * min_offset)
    end_time = T0 + timedelta(minutes=30 * max_offset)
    timestamps = pd.date_range(start=start_time, periods=max_offset - min_offset, freq='30min')

    # Plot only Hp30
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, target, lw=1, label='Hp30')

    # Vertical line at T0
    ax.vlines(T0, 0, max(target) + 1, colors='black', linestyles='dashed', label='T0')

    # Highlight forecast window
    ax.axvspan(F_start, F_end, color='grey', alpha=0.3)

    # Max Hp30
    ax.hlines(max_target, F_start, F_end, colors='red', label=f'max Hp30={max_target:.2f}')

    # Storm threshold
    ax.hlines(4.66, timestamps[0], timestamps[-1], colors='dodgerblue', 
              label='Storm Threshold=4.66', linestyles='dotted')

    # Persistence forecast
    if persistence is not None:
        ax.hlines(persistence, timestamps[48], timestamps[-1], 
                  colors='green', linestyles='dashdot', 
                  label=f'Persistence={persistence:.2f}', lw=2)

    # Formatting
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(0, max(target) + 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    ax.set_xlabel("Time")
    ax.set_ylabel("Hp30")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_title(
        f"Hp30 Persistence Forecast Around T0 = {T0}\n",
        fontsize=14
    )

    plt.show()



def plot_quadrant(ax1, ax2, window, min_offset, max_offset, forecast=None, persistence=None, twentyseven=None):
    v = window['v']
    omni = window['omni_plotting']
    target = window['target_plotting']
    max_target = window['max_target']
    T0 = window['T0']
    F_start = window['target_start']
    F_end = window['target_end']

    correct = None
    if forecast is not None:
        correct = int(forecast + 0.5) == int(max_target > 4.66)

    # Create time axis
    start_time = T0 + timedelta(minutes=30 * min_offset)
    end_time = T0 + timedelta(minutes=30 * max_offset)
    timestamps = pd.date_range(start=start_time, periods=max_offset - min_offset, freq='30min')

    # --- Top panel ---
    for i in range(v.shape[1]):
        label = 'v_ensemble' if i == 0 else None
        ax1.plot(timestamps, v[:, i], 'blue', lw=1, alpha=0.1, label=label)
    ax1.plot(timestamps, omni, 'k', lw=1.5, label='OMNI')
    ax1.vlines(T0, min((min(v.flatten()), min(omni))) - 50,
               max((max(v.flatten()), max(omni))) + 50,
               colors='black', linestyles='dashed', label='$T_0$')

    # --- Bottom panel ---
    ax2.plot(timestamps, target, '#E66100', lw=1.5, label='Hp30')
    ax2.vlines(T0, 0, max(max(target)+1, 6),
               colors='black', linestyles='dashed', label='$T_0$')

    # Max Hp30 marker
    target_df = pd.DataFrame({"time": timestamps, "target": target})
    window_df = target_df[(target_df["time"] >= F_start) & (target_df["time"] <= F_end)]
    if not window_df.empty:
        idx_max = window_df["target"].idxmax()
        max_time = window_df.loc[idx_max, "time"]
        max_value = window_df.loc[idx_max, "target"]
        ax2.scatter(max_time, max_value, color="royalblue", s=80, zorder=5,
                    label=f"max Hp30={max_value:.2f}")

    ax2.hlines(4.66, timestamps[0], timestamps[-1], colors='#D41159',
               label='Storm Threshold=4.66', linestyles='dashed')

    # Background colour depending on correctness
    GREEN = (0.6, 0.9, 0.6, 0.5)
    RED = (0.9, 0.6, 0.6, 0.5)
    bg_colour = 'white' if correct is None else (GREEN if correct else RED)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.axvspan(F_start, F_end, color='grey', alpha=0.4)
        ax.set_facecolor(bg_colour)
        ax.tick_params(labelsize=18)
        ax.grid(True)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('center')

    ax1.set_ylim(min((min(v.flatten()), min(omni))) - 50, max((max(v.flatten()), max(omni))) + 50)
    ax2.set_ylim(0, max(max(target)+1, 6))
    ax2.set_xlabel("Time", fontsize=18)
    ax1.set_ylabel('v (km/s)', fontsize=18)
    ax2.set_ylabel('Hp30', fontsize=18)

    if forecast is not None:
        ax1.set_title(f"Weighted Mean={forecast:.3f}, Persistence={float(persistence)}, 27-day Recurrence={float(twentyseven)}",
                      fontsize=18, pad=20)


    

    