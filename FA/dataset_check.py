from pathlib import Path
import numpy as np
from model_architecture import nc4PyDataset

use_time_info = False #True
use_without_time_info_data = False

max_wind_channel = 4 if not use_without_time_info_data else 2
meta_channnel = None if use_time_info or use_without_time_info_data else slice(2, None)


raw_data_path = Path('TC_data_GeoSciAI2024/')
nc4_data_path = raw_data_path / 'concatenated_data_with_timeinfo' if not use_without_time_info_data else raw_data_path / 'concatenated_data'
stats_path = nc4_data_path / 'stats'

tp_list = list(nc4_data_path.glob('*.nc'))
train = []
test = []
out_of_sample = []
for tp in tp_list:
    year = int(tp.stem[:4])
    if year >= 2003:
        out_of_sample.append(tp)
    elif year >= 2000:
        test.append(tp)
    else:
        train.append(tp)

train_dataset = nc4PyDataset(train, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)
val_dataset = nc4PyDataset(test, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)
eval_dataset = nc4PyDataset(out_of_sample, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)


for dataset in [train_dataset, val_dataset, eval_dataset]:
    data = []
    for i in range(len(dataset)):
        data.append((dataset[i][1]))
    y_true = np.concatenate(data)
    print(y_true.shape)