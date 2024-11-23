
import GPy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime
from datetime import timedelta
import random
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sklearn.preprocessing
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from GPy.kern import DiffKern
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from mpl_toolkits.mplot3d import Axes3D
############################################################
font = FontProperties(family='Arial')
plt.rcParams['font.family'] = font.get_name()

kernel_list = ['RBF','Matern52','Matern32','OU','Exponential','ExpQuad']
kername = 0
dict = {'building':2,
        'floor': 7,
        'kername':kernel_list[kername],
        'variance':1,
        'ratio':1, # source/aug
        'sigma':0.05, # Noise of observations
        'sigma_der':0.05, # Noise of derivative observations
        'mode':0, # 0 by fllor, 1 by neigh, 2 by building,
        'write_the_file':1 # None for not write the file
         }
dict_new = {'building':2,
        'floor': 7,
        'kername':kernel_list[kername],
        'variance':1,
        'ratio':1, # source/aug
        'sigma':0.05, # Noise of observations
        'sigma_der':0.05, # Noise of derivative observations
        'mode':0, # 0 by fllor, 1 by neigh, 2 by building,
        'write_the_file':1 # None for not write the file
         }
def kern_sel(kername):
    if kername == 0:
        se = GPy.kern.RBF(input_dim = 3, lengthscale=10, variance = dict_new['variance'])
    if kername == 1:
        se = GPy.kern.Matern52(input_dim= 3, lengthscale= 10, variance = dict_new['variance'])
    if kername == 2:
        se = GPy.kern.Matern32(input_dim= 3, lengthscale= 10, variance = dict_new['variance'])
    if kername == 3:
        se = GPy.kern.OU(input_dim= 3, lengthscale=10, variance= dict_new['variance'])
    if kername == 4:
        se = GPy.kern.Exponential(input_dim= 3, lengthscale=10, variance= dict_new['variance'])
    if kername == 5:
        se = GPy.kern.ExpQuad(input_dim= 3, lengthscale=10, variance= dict_new['variance'])

    return se


def find_min_max_diff(df1, df2):
    # 创建一个 DataFrame 来存储差值的绝对值
    diff_df = pd.DataFrame()
    # 计算两个 DataFrame 对应列的差值的绝对值
    for i in range(1, 194):
        col_name = f"WAP{i:03d}"
        diff_df[col_name] = np.abs(df1[col_name] - df2[col_name])
    #   diff_df[col_name] = np.abs(df1[col_name] - df2[col_name])
    print(diff_df)
    # 计算最小和最大差值
    min_diff = diff_df.min().min()
    max_diff = diff_df.max().max()

    # 找到最小和最大差值对应的 WAP
    min_wap = diff_df.min(axis=1).idxmin()
    max_wap = diff_df.max(axis=1).idxmax()
    # 获取最小和最大差值对应的WAP编号
    minWap = diff_df.columns[diff_df.loc[min_wap] == min_diff][0]
    maxWap = diff_df.columns[diff_df.loc[max_wap] == max_diff][0]

    return min_diff, max_diff, minWap, maxWap, diff_df



plt.rcParams['font.family'] = font.get_name()

kernel_list = ['RBF','Matern52','Matern32','OU','Exponential','ExpQuad']
kername = 0
dict = {'building':2,
        'floor': 7,
        'kername':kernel_list[kername],
        'variance':1,
        'ratio':1, # source/aug
        'sigma':0.05, # Noise of observations
        'sigma_der':0.05, # Noise of derivative observations
        'mode':0, # 0 by fllor, 1 by neigh, 2 by building,
        'write_the_file':1 # None for not write the file
         }
dict_new = {'building':2,
        'floor': 7,
        'kername':kernel_list[kername],
        'variance':1,
        'ratio':1, # source/aug
        'sigma':0.05, # Noise of observations
        'sigma_der':0.05, # Noise of derivative observations
        'mode':0, # 0 by fllor, 1 by neigh, 2 by building,
        'write_the_file':1 # None for not write the file
         }
def kern_sel(kername):
    if kername == 0:
        se = GPy.kern.RBF(input_dim = 3, lengthscale=10, variance = dict_new['variance'])
    if kername == 1:
        se = GPy.kern.Matern52(input_dim= 3, lengthscale= 10, variance = dict_new['variance'])
    if kername == 2:
        se = GPy.kern.Matern32(input_dim= 3, lengthscale= 10, variance = dict_new['variance'])
    if kername == 3:
        se = GPy.kern.OU(input_dim= 3, lengthscale=10, variance= dict_new['variance'])
    if kername == 4:
        se = GPy.kern.Exponential(input_dim= 3, lengthscale=10, variance= dict_new['variance'])
    if kername == 5:
        se = GPy.kern.ExpQuad(input_dim= 3, lengthscale=10, variance= dict['variance'])

    return se


def find_min_max_diff32(df1, df2):
    # 创建一个 DataFrame 来存储差值的绝对值
    diff_df32 = pd.DataFrame()
    # 计算两个 DataFrame 对应列的差值的绝对值
    for i in range(1, 124):
        col_name = f"WAP{i:03d}"
        diff_df32[col_name] = np.abs(df1[col_name] - df2[col_name])
    #   diff_df[col_name] = np.abs(df1[col_name] - df2[col_name])
    # 计算最小和最大差值
    min_diff32 = diff_df32.min().min()
    max_diff32 = diff_df32.max().max()

    # 找到最小和最大差值对应的 WAP
    min_wap32 = diff_df32.min(axis=1).idxmin()
    max_wap32 = diff_df32.max(axis=1).idxmax()
    # 获取最小和最大差值对应的WAP编号
    minWap32 = diff_df32.columns[diff_df32.loc[min_wap32] == min_diff32][0]
    maxWap32 = diff_df32.columns[diff_df32.loc[max_wap32] == max_diff32][0]

    return min_diff32, max_diff32, minWap32, maxWap32, diff_df32


source_data_path = "/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/train_data.csv"
fake_data_path = "/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/train_data_fake_testwang.csv"
test_data_path = "/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/test_data.csv"
with open(source_data_path, "r") as _file:
    df_raw = pd.read_csv(_file).loc[:, "WAP001":"y"]
df_raw = df_raw.drop('building', axis=1)
df_raw = df_raw.drop('device', axis=1)
df_raw["x"] = pd.to_numeric(df_raw["x"], errors='coerce')
# 将 time 列转换为日期时间格式，其中每个值都是一个日期时间对象
df_raw["time"] = pd.to_datetime(df_raw["time"].apply(lambda t_str: datetime.strptime(f'2023-{t_str}', '%Y-%m-%d-%H-%M')))
df_raw["y"] = pd.to_numeric(df_raw["y"], errors='coerce')
df_raw["floor"] = pd.to_numeric(df_raw["floor"], errors='coerce')
df = df_raw.groupby(['x', 'y'], as_index=False).mean()
xy = df.loc[:, "x":"y"].to_numpy()
z = df.loc[:, "WAP001": "WAP466"].to_numpy()
# 标准化有助于模型更好地拟合数据better fit data，去掉会帮助模型更好地处理不同量级的特征  handle features of different magnitudes.
# standarder = sklearn.preprocessing.StandardScaler()
# standarder.fit(z)
# z = standarder.transform(z)

# 过滤掉所有值为 -110 的 AP 列
valid_columns = ~np.all(z == -110, axis=0)
# 获取有效的 WAP 列名
wap_columns = df.columns[df.columns.get_loc("WAP001"):df.columns.get_loc("WAP466") + 1]
valid_waps = wap_columns[valid_columns]

# 只取前10个有效的 WAP 列
valid_waps = valid_waps[:3]
z1 = z[:, :3]

# 定义 ADF 检验函数
def perform_adf_test(series, name):
    if np.all(series == series[0]):
        return
    result = adfuller(series)
    print(f'ADF Test for {name}')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    print()

# 定义绘制 ACF 和 PACF 图的函数
# def plot_acf_pacf(series, name):
#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 1, 1)
#     plot_acf(series, lags=min(40, len(series) // 2 - 1), ax=plt.gca())
#     plt.title(f'ACF Plot for {name}')
#
#     plt.subplot(2, 1, 2)
#     plot_pacf(series, lags=min(40, len(series) // 2 - 1), ax=plt.gca())
#     plt.title(f'PACF Plot for {name}')
#     plt.tight_layout()
#     plt.show()
#
# # 对每个 AP 的 RSSI 数据进行平稳性检验并绘制 ACF 和 PACF 图
# for i, ap_name in enumerate(valid_waps):
#     ap_data = z1[:, i]
#     if ap_data.size == 0 or np.isnan(ap_data).all():
#         continue
#     # ADF 检验
#     perform_adf_test(ap_data, ap_name)
#     # 绘制 ACF 和 PACF 图
#     plot_acf_pacf(ap_data, ap_name)
############################################################
#likelihood function
gauss = GPy.likelihoods.Gaussian(variance=dict['sigma']**2)
gauss_der = GPy.likelihoods.Gaussian(variance=dict['sigma_der']**2)
############################################################
#Create the model
k = kern_sel(kername)
se_der = GPy.kern.DiffKern(k, 0)
# se_der_new = GPy.kern.DiffKern(k_new, 0)
m = GPy.models.MultioutputGP(X_list=[xy], Y_list=[z], kernel_list=[k, se_der], likelihood_list=[gauss, gauss_der])
fdata_num = int(len(df_raw) / dict['ratio'])
all_x = df_raw['x'].values
all_y = df_raw['y'].values
half_all_x = all_x / 1.1
half_all_y = all_y / 1.1

xy_pred = np.array([[x, y, 0] for x, y in zip(half_all_x, half_all_y)])
mu, var = m.predict_noiseless(Xnew=xy_pred)
df_new = pd.DataFrame(mu, columns=df_raw.columns[:466]).astype("int64")
df_new["x"] = xy_pred[:, 0]
df_new["y"] = xy_pred[:, 1]
df_new["floor"] = dict['floor']
df_new["building"] = dict['building']
df_new["time"] = np.zeros(xy_pred[:, 0].shape, dtype=int)

with open(fake_data_path, "w") as _file:
     _file.write(df_new.to_csv(index=False))
print('Data augmented file writes')

wap_column = 'WAP004'
source_data_df = pd.read_csv(source_data_path)
fake_data_df = pd.read_csv(fake_data_path)
testing_data = pd.read_csv(test_data_path)
source_x = source_data_df['x'].values
source_y = source_data_df['y'].values

t = np.array(df_raw['time'].tolist())
# date_time_objects = pd.to_datetime(t,errors='coerce')
xy_time_pred = np.stack((all_x, all_y, t), axis=1)
xy_time_pred_3d = xy_time_pred[:, :, np.newaxis]

t1 = np.array([datetime.now() for _ in range(735)])
t_numeric = [t_val.timestamp() for t_val in t1]
#将时间格式转换为将时间转换为数字形式，这里使用时间戳（秒）,将原来时间戳形式的时间，使用SCAscaling
numeric_time = df_raw["time"].astype(int) // 10**9
numeric_time_series = pd.Series(numeric_time)
scaler = MinMaxScaler(feature_range=(0, 1))
# 将时间数据转换为 [0, 1] 范围
scaled_time_series = scaler.fit_transform(numeric_time.values.reshape(-1, 1)).flatten()
scaled_time_series_rounded = scaled_time_series.round(2)  # 保留两位小数
# print(scaled_time_series_rounded)
scaled_time_series_float= scaled_time_series_rounded.tolist()



# 时间戳 一维数组
t_discrete1 = np.array(t_numeric)
# 将离散类别数组与原始数组合并
xy_time_pred_discrete = np.concatenate((xy_time_pred, t_discrete1.reshape(-1, 1)), axis=1)
t_discrete2 = xy_time_pred_discrete[:, -1]
t_discrete2_str = [str(time) for time in t_discrete2]
# 使用字符串切片功能删除每个时间戳中的 '1725949898.' 部分
modified_times = [time[14:] for time in t_discrete2_str]
modified_times_new = [float(time) for time in modified_times if time != '']
modified_times_new1=np.array(modified_times_new)
xy_time_pred_discrete1 = np.concatenate((xy_time_pred, modified_times_new1.reshape(-1, 1)), axis=1)
xy_time_pred_discrete11 = xy_time_pred_discrete1[:, :, np.newaxis]
x_range = (xy_time_pred_discrete11[:, 0].min(), xy_time_pred_discrete11[:, 0].max())
y_range = (xy_time_pred_discrete11[:, 1].min(), xy_time_pred_discrete11[:, 1].max())
timestamp_range = (xy_time_pred_discrete11[:, 2].min(), xy_time_pred_discrete11[:, 2].max())
time_range = (xy_time_pred_discrete11[:, 3].min(), xy_time_pred_discrete11[:, 3].max())


start_column = 'WAP001'
end_column = 'WAP466'
# 提取这些列
z1 = df_raw.loc[:, start_column:end_column].values
cols_to_keep = np.any(z1 != -110, axis=0)
z1_filtered = z1[:, cols_to_keep]
xy_time_train2 = np.stack((all_x, all_y, scaled_time_series_rounded), axis=1)

fake_data_path_new = "/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/new_train_data_fake_testwang.csv"
fake_data_path_new32 = "/Users/bluewang/Desktop/kk.csv"
# rbf
k_new = kern_sel(kername == 0)
se_der_new = GPy.kern.DiffKern(k_new, 0)
m1 = GPy.models.MultioutputGP(X_list=[xy_time_train2], Y_list=[z1_filtered], kernel_list=[k_new, se_der_new], likelihood_list=[gauss, gauss_der])

zeros_column = np.zeros((xy_time_train2.shape[0], 1))  # 创建一个与行数相同，元素全为0的列
xy_time_pred2_with_zeros = np.hstack((xy_time_train2, zeros_column))  # 将这个列添加到原始数组中
mu1, var1 = m1.predict_noiseless(Xnew=xy_time_pred2_with_zeros)

df_new1 = pd.DataFrame(mu1, columns=df_raw.columns[:193]).astype("int64")
df_new1["x"] = xy_time_pred2_with_zeros[:, 0]
df_new1["y"] = xy_time_pred2_with_zeros[:, 1]
df_new1["floor"] = dict['floor']
df_new1["building"] = dict['building']
df_new1["time"] = pd.to_datetime(df_raw['time'])
df_new1["timestamp"] = xy_time_pred2_with_zeros[:, 2]
#优化
m1.optimize(messages=1, ipython_notebook=False)
with open(fake_data_path_new, "w") as _file:
    _file.write(df_new1.to_csv(index=False))

fake_data_path_new_df = pd.read_csv(fake_data_path_new)
fake_data_path_new_df32 = pd.read_csv(fake_data_path_new32)

# 剪辑函数，用于剪辑不在 (-110, 0) 范围内的值
def clip_values(val):
    if val < -110 or val > 0:
        return -110
    return val
# 定义要剪辑的列的范围
start_col = 'WAP001'
end_col = 'WAP193'
# 确保列名在数据帧中存在
if start_col in fake_data_path_new_df.columns and end_col in fake_data_path_new_df.columns:
    # 获取列的索引
    start_idx = fake_data_path_new_df.columns.get_loc(start_col)
    end_idx = fake_data_path_new_df.columns.get_loc(end_col) + 1  # 加1是因为 slice 是左闭右开的区间
    # 剪辑指定列的值
    for column in fake_data_path_new_df.columns[start_idx:end_idx]:
        fake_data_path_new_df[column] = fake_data_path_new_df[column].apply(clip_values)
    # CSV 文件
    output_filename = fake_data_path_new
    fake_data_path_new_df.to_csv(output_filename, index=False)
else:
    print(f"Error: One or both of the specified columns '{start_col}' and '{end_col}' do not exist in the dataframe.")


# m32
start_idx = fake_data_path_new_df32.columns.get_loc(start_col)
end_idx = fake_data_path_new_df32.columns.get_loc(end_col) + 1  # 加1是因为 slice 是左闭右开的区间
    # 剪辑指定列的值
for column in fake_data_path_new_df32.columns[start_idx:end_idx]:
    fake_data_path_new_df32[column] = fake_data_path_new_df32[column].apply(clip_values)
    # CSV 文件
output_filename32 = fake_data_path_new32
fake_data_path_new_df32.to_csv(output_filename32, index=False)


# 调用函数
min_diff, max_diff, minWap, maxWap, diff_df = find_min_max_diff(df_raw, fake_data_path_new_df)
print(f"最小差值： {min_diff}, 对应WAP: {minWap}")
print(f"最大差值： {max_diff}, 对应WAP: {maxWap}")
error_df = pd.DataFrame(diff_df)
average_error_from_diff_df = np.mean(error_df)
print(average_error_from_diff_df)

min_diff32, max_diff32, minWap32, maxWap32, diff_df32 = find_min_max_diff32(df_raw, fake_data_path_new_df32)
# print(f"最小差值： {min_diff32}, 对应WAP: {minWap32}")
# print(f"最大差值： {max_diff32}, 对应WAP: {maxWap32}")

df_raw['time'] = pd.to_datetime(df_raw['time'])
# 找到差值的最大值
max_diffs = diff_df.max()
# # 创建一个图表，其中包含每个WAP的最大误差
plt.figure(figsize=(15, 10))
y = diff_df['WAP001']
x = df_raw['time']
times = mdates.date2num(x)
# Create an array of the same length as x data
x_extended = np.arange(len(x))
# Plot the data with the extended x data
plt.plot(times, y,label=f"WAP001- Difference")
plt.plot(color='grey', linestyle='--')  # 绘制水平线表示差值为0
# 在差值点上添加圆点
for time, diff in zip(times, y):
    plt.scatter(time, diff, color='red', s=20, marker='o')
plt.legend()
# 设置 x 轴的格式化器为日期时间格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
# 旋转日期标记以避免重叠
plt.gcf().autofmt_xdate()
plt.xlabel("Time")
plt.ylabel("Difference")
plt.tight_layout()
plt.show()



errors1 = diff_df['WAP001'].dropna()  # 确保没有NaN值
errors2 = diff_df32['WAP001'].dropna()  # 确保没有NaN值
# 对两个误差值数组进行排序
sorted_errors1 = np.sort(errors1)
sorted_errors2 = np.sort(errors2)

# 计算两个CDF的值
# 蓝色是rbf 红色是Matern52  剩下的都和它重合
cdf_values1 = np.arange(1, len(sorted_errors1) + 1) / len(sorted_errors1)
cdf_values2 = np.arange(1, len(sorted_errors2) + 1) / len(sorted_errors2)
plt.figure(figsize=(10, 6))
plt.plot(sorted_errors1, cdf_values1, marker='o', linestyle='-', color='b', label='CDF of Errors1', markersize=1, linewidth=0.5)
plt.plot(sorted_errors2, cdf_values2, marker='x', linestyle='-', color='r', label='CDF of Errors2', markersize=1, linewidth=0.5)
plt.title('CDF Comparison of WAP001 Errors')
plt.xlabel('Error Value')
plt.ylabel('CDF')
plt.legend()  # 显示图例
plt.grid(True)
plt.show()



# Coordinates to filter the data     随记中代表的 图一Signal Strength Over Time at Locations (x1 = 9.3, y1 = 2.4)
x1 = 9.3
y1 = 2.4
train_data_filtered = df_raw[(df_raw['x'] == x1) & (df_raw['y'] == y1)]
# fake_data_filtered = fake_data_path_new_df[(fake_data_path_new_df['x'] == x1/1.1) & (fake_data_path_new_df['y'] == y1/1.1)]
fake_data_filtered = fake_data_path_new_df[(fake_data_path_new_df['x'] == x1) & (fake_data_path_new_df['y'] == y1)]
common_indices = train_data_filtered.index.intersection(df_raw.index)
train_data_filtered = train_data_filtered.loc[common_indices]
df_raw = df_raw.loc[common_indices]

common_indices_fake = fake_data_filtered.index.intersection(fake_data_path_new_df.index)
fake_data_filtered = fake_data_filtered.loc[common_indices_fake]
fake_data_path_new_df = fake_data_path_new_df.loc[common_indices_fake]
# 提取时间列
train_times = df_raw.loc[common_indices, 'time']
fake_times = fake_data_path_new_df.loc[common_indices_fake, 'time']
train_times_num = mdates.date2num(train_times)
fake_times_num = mdates.date2num(fake_times)
plt.figure(figsize=(15, 7))
# Plot for train_data
plt.plot(train_times_num, train_data_filtered['WAP001'], label='Train Data', color='blue')
# Plot for fake_data
plt.plot(fake_times_num, fake_data_filtered['WAP001'], label='Augmented Data', color='red', linestyle='--')
# 设置 x 轴的格式化器为日期时间格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
# 旋转日期标记以避免重叠
plt.gcf().autofmt_xdate()
plt.xlabel('Time')
plt.ylabel('RSSI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


