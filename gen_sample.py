import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def sample_user_candidates(group, n):
    online_items = group[np.negative(group['time_ms'].isna())]
    virtual_items = group[group['time_ms'].isna()]
    if len(online_items) > n:
        sample_items = online_items.sample(n)
    else:
        sample_items = pd.concat([online_items, virtual_items.sample(n-len(online_items))])
    return sample_items

parser = argparse.ArgumentParser("AAAI-2023...")
parser.add_argument("--data_path", type=str, default="KuaiRand-1K")
parser.add_argument("--split_date", type=str, default="2022-05-05")
parser.add_argument("--user_sample_size", type=int, default=10000)

args = parser.parse_args()
data_path = args.data_path
split_date = args.split_date
user_sample_size = args.user_sample_size

data_type = data_path.split('-')[-1].lower()
nrows = 1e12
df1 = pd.read_csv(os.path.join(data_path, f"data/log_standard_4_08_to_4_21_{data_type}.csv"), nrows=nrows, parse_dates=['date'])
df2 = pd.read_csv(os.path.join(data_path, f"data/log_standard_4_22_to_5_08_{data_type}.csv"), nrows=nrows, parse_dates=['date'])

data = pd.concat([df1, df2], axis=0, ignore_index=True)
data = data[(data['tab'] == 1) & (data['play_time_ms']) > 0]
data.drop(['tab'], axis=1, inplace=True)
data = data[data['duration_ms'] > 0]
data['duration_s'] = data['duration_ms'] / 1000
data['play_time_s'] = data['play_time_ms'] / 1000
data['play_rate'] = data['play_time_ms'] / data['duration_ms']
data = data[data['play_rate'] <= 5]

train = data[data['date'] <= split_date][['user_id', 'video_id']].reset_index(drop=True)
test = data[data['date'] > split_date][['user_id', 'video_id', 'time_ms']].reset_index(drop=True)

output_data = []

for k in range(0, 10):
    candidates = pd.read_pickle(f"./results/fullrank_{data_type}_{k}.pkl")
    user_ids = candidates['user_id'].unique()
    for i in tqdm(user_ids):

        # 剔除训练集曝光样本
        user_sample = candidates[candidates['user_id']==i].merge(train[train['user_id']==i], on=['user_id', 'video_id'], how='left', indicator=True)
        user_sample = user_sample[user_sample['_merge'] == 'left_only']
        user_sample = user_sample.drop(['_merge'], axis=1)
        # 提取测试集曝光样本
        user_sample = user_sample.merge(test[test['user_id']==i], on=['user_id', 'video_id'], how='left')
        # print(user_sample.columns)
        # 对用户进行采样，构建候选集
        online_items = user_sample[np.negative(user_sample['time_ms'].isna())]
        if len(online_items) < 10:
            continue
        virtual_items = user_sample[user_sample['time_ms'].isna()]
        if len(online_items) >= user_sample_size:
            sample_items = online_items.sample(user_sample_size)
        else:
            # print(user_sample_size, len(online_items), user_sample_size - len(online_items))
            virtual_items = virtual_items.sample(user_sample_size - len(online_items))
            sample_items = pd.concat([online_items, virtual_items])
        output_data.append(sample_items)

output_data = pd.concat(output_data, axis=0, ignore_index=True)
output_data.to_pickle('./sample_data.pkl')

# 初始化数据
base_date = pd.to_datetime('2022-05-09')
items = pd.read_csv("KuaiRand-1K/data/video_features_basic_1k.csv")
items = items[np.negative(items['upload_dt'].isna())]
items['photo_age'] = (base_date - pd.to_datetime(items['upload_dt'], format="%Y-%m-%d", errors='coerce')).dt.days
items['freshness'] = 1 / items['photo_age']

df = pd.read_pickle('sample_data.pkl')
df = df[df['play_time_s'] > 0]
df = df[df['is_like'] > 0]
df = df[df['is_follow'] > 0]
df = df[df['is_comment'] > 0]
df = df[df['is_forward'] > 0]
df = df.merge(items, on='video_id', how='inner')

def log_minmax_norm(col):
    tmp = np.log(col)
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    return tmp

df['pt_norm'] = log_minmax_norm(df["play_time_s"])
df['like_norm'] = log_minmax_norm(df["is_like"])
df['follow_norm'] = log_minmax_norm(df['is_follow'])
df['comment_norm'] = log_minmax_norm(df['is_comment'])
df['forward_norm'] = log_minmax_norm(df['is_forward'])

df = df[['user_id', 'freshness', 'pt_norm', 'like_norm', 'follow_norm', 'comment_norm', 'forward_norm']]
df.to_pickle("data_for_plots_v2.pkl")
