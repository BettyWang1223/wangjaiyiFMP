
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


t = np.array(df_raw['time'].tolist())
#将时间格式转换为将时间转换为数字形式，这里使用时间戳（秒）,将原来时间戳形式的时间，使用SCAscaling
numeric_time = df_raw["time"].astype(int) // 10**9
numeric_time_series = pd.Series(numeric_time)
scaler = MinMaxScaler(feature_range=(0, 1))
# 将时间数据转换为 [0, 1] 范围
scaled_time_series = scaler.fit_transform(numeric_time.values.reshape(-1, 1)).flatten()
scaled_time_series_rounded = scaled_time_series.round(2)  # 保留两位小数


# 预测时间
scaled_time_series_increased1 = scaled_time_series_rounded + 0.01
date_str = '2023/7/1'
date_format = '%Y/%m/%d'
timestamp = int(datetime.strptime(date_str, date_format).timestamp())
#
# def reverse_scale(scaled_value, scaler, original_time_stamps):
#     min_value = original_time_stamps.min()
#     max_value = original_time_stamps.max()
#     original_value = scaled_value * (max_value - min_value) + min_value
#     return original_value
#
# # 示例：找到增加后的值0.04对应的时间
# original_time_stamps = numeric_time_series.values
# original_timestamp = reverse_scale(scaled_time_series_increased1, scaler, original_time_stamps)
# # 将时间戳转换回时间格式
# original_time = pd.to_datetime(original_timestamp, unit='s')

start_column = 'WAP001'
end_column = 'WAP466'
# 提取这些列
z1 = df_raw.loc[:, start_column:end_column].values
cols_to_keep = np.any(z1 != -110, axis=0)
z1_filtered = z1[:, cols_to_keep]

Q=pd.to_datetime(df_raw["time"], format="%Y/%m/%d %H:%M")
Q = Q.astype(int) // 10**9
t_discrete2_str = [str(time) for time in Q]
modified_times = [time[3:] for time in t_discrete2_str]
modified_times_new = [float(time) for time in modified_times if time != '']
modified_times_new1=np.array(modified_times_new)
scaler1 = MinMaxScaler(feature_range=(0, 1))
# 将时间数据转换为 [0, 1] 范围
scaled_time_series1 = scaler1.fit_transform(Q.values.reshape(-1, 1)).flatten()
scaled_time_series_rounded1 = scaled_time_series1.round(2)  # 保留两位小数

P=pd.to_datetime(df_raw["time"], format="%Y/%m/%d %H:%M")+pd.Timedelta(days=1)
P = P.astype(int) // 10**9
t_discrete2_str1 = [str(time) for time in P]
modified_times1 = [time[3:] for time in t_discrete2_str1]
modified_times_new3 = [float(time) for time in modified_times1 if time != '']
modified_times_new2=np.array(modified_times_new3)

xy_time_train2 = np.stack((all_x, all_y, scaled_time_series_rounded), axis=1)
xy_time_train2_pred = np.stack((all_x, all_y, modified_times_new2), axis=1)

fake_data_path_new = "/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/new_train_data_fake_testwang.csv"
# rbf
k_new = kern_sel(kername == 0)
se_der_new = GPy.kern.DiffKern(k_new, 0)
m1 = GPy.models.MultioutputGP(X_list=[xy_time_train2], Y_list=[z1_filtered], kernel_list=[k_new, se_der_new], likelihood_list=[gauss, gauss_der])


X1=xy_time_train2_pred[:35,]
zeros_column = np.zeros((xy_time_train2.shape[0], 1))  # 创建一个与行数相同，元素全为0的列
xy_time_pred2_with_zeros = np.hstack((xy_time_train2, zeros_column))  # 将这个列添加到原始数组中
mu1, var1 = m1.predict_noiseless(Xnew=xy_time_pred2_with_zeros)

df_new1 = pd.DataFrame(mu1, columns=df_raw.columns[:193]).astype("int64")
df_new1["x"] = xy_time_pred2_with_zeros[:, 0]
df_new1["y"] = xy_time_pred2_with_zeros[:, 1]
df_new1["time"] = pd.to_datetime(df_raw["time"])
df_new1["scaler_time"] = xy_time_pred2_with_zeros[:, 2]
#优化
m1.optimize(messages=1, ipython_notebook=False)
with open(fake_data_path_new, "w") as _file:
    _file.write(df_new1.to_csv(index=False))
fake_data_path_new_df = pd.read_csv(fake_data_path_new)

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



# 调用函数
min_diff, max_diff, minWap, maxWap, diff_df = find_min_max_diff(df_raw, fake_data_path_new_df)
# print(f"最小差值： {min_diff}, 对应WAP: {minWap}")
# print(f"最大差值： {max_diff}, 对应WAP: {maxWap}")
error_df = pd.DataFrame(diff_df)
average_error_from_diff_df = np.mean(error_df)
print(average_error_from_diff_df)

# 提取 x 和 y 列
x = xy_time_train2[:, 0]
y = xy_time_train2[:, 1]
x1 = x[:34,]
y1 = y[:34,]
# 计算 x 和 y 的总偏移量
x_offset = x1[-1] - x1[0]
y_offset = y1[-1] - y1[0]

print(f"x 的总偏移量: {x_offset}")
print(f"y 的总偏移量: {y_offset}")


data_list = []
data_list.append(fake_data_path_new_df)
# 将所有DataFrame合并为一个DataFrame
data = pd.concat(data_list, ignore_index=True)


# 假设你的数据集的日期列名为'date'
data['scaler_time'] = data['scaler_time'].astype(str)
train_data = data.loc[data['scaler_time'] <= '0.47']
test_data = data.loc[data['scaler_time'] > '0.47']
train_data.to_csv('/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/new_train.csv', index=False)
test_data.to_csv('/Users/bluewang/code-FMP/GP_indoorLoc/database/XJTLU Dynamic/timeview_floor7/new_test.csv', index=False)
