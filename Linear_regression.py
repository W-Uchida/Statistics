import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import ipywidgets as widgets
from IPython.display import display

# データの読み込み
def load_default_data():
    diabetes = load_diabetes()
    x = diabetes.data[:, 0]  # age
    y = diabetes.data[:, 2]  # average blood pressure
    return x, y

def read_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df.iloc[:, 0], df.iloc[:, 1]

def calculate_rss(slope, intercept, x, y):
    predictions = slope * x + intercept
    rss = np.sum((y - predictions) ** 2)
    return rss

def update_distances(slope, intercept, x, y):
    distances = []
    for xi, yi in zip(x, y):
        yi_pred = slope * xi + intercept
        distances.append((xi, yi, yi_pred))
    return distances

def plot_data(x, y, slope=None, intercept=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Data', s=15)
    if slope is not None and intercept is not None:
        ax.plot(x, slope * x + intercept, color='red', label=f'Fit: y={slope:.2f}x + {intercept:.2f}\nRSS={calculate_rss(slope, intercept, x, y):.2f}')
        distances = update_distances(slope, intercept, x, y)
        for (xi, yi, yi_pred) in distances:
            ax.plot([xi, xi], [yi, yi_pred], color='blue', linestyle='dotted', linewidth=0.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Blood Pressure')
    ax.set_title('Interactive Regression Line Fit')
    ax.legend()
    ax.grid(True)
    plt.show()

# 初期データの読み込み
x, y = load_default_data()
init_slope, init_intercept = np.polyfit(x, y, 1)

# スライダーの設定
slope_slider = widgets.FloatSlider(value=init_slope, min=init_slope-10, max=init_slope+10, step=0.01, description='Slope:')
intercept_slider = widgets.FloatSlider(value=init_intercept, min=init_intercept-50, max=init_intercept+50, step=0.1, description='Intercept:')

def on_update(change):
    slope = slope_slider.value
    intercept = intercept_slider.value
    plot_data(x, y, slope, intercept)

# スライダーのイベントを登録
slope_slider.observe(on_update, names='value')
intercept_slider.observe(on_update, names='value')

# 最初のプロット
plot_data(x, y, init_slope, init_intercept)

# スライダーを表示
display(slope_slider, intercept_slider)

# リセットボタン
reset_button = widgets.Button(description='Reset')
def on_reset_button_clicked(b):
    slope_slider.value = init_slope
    intercept_slider.value = init_intercept

reset_button.on_click(on_reset_button_clicked)
display(reset_button)
