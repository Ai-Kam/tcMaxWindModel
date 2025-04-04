import netCDF4 as nc4
import numpy as np
from pathlib import Path

use_time_info = False

raw_data_path = Path('TC_data_GeoSciAI2024/')

with_time_nc4_data_path = raw_data_path / 'concatenated_data_with_timeinfo'
without_time_nc4_data_path = raw_data_path / 'concatenated_data'
with_time_stats_path = with_time_nc4_data_path / 'stats'
without_time_stats_path = without_time_nc4_data_path / 'stats'

tp_list = list(with_time_nc4_data_path.glob('*.nc'))

stats_with_time = np.load(with_time_stats_path / 'time_stats.npz')
time_mean_with_time = stats_with_time['time_mean']
time_std_with_time = stats_with_time['time_std']
stats_without_time = np.load(without_time_stats_path / 'time_stats.npz')
time_mean_without_time = stats_without_time['time_mean']
time_std_without_time = stats_without_time['time_std']
print(time_std_with_time[2:] - time_std_without_time)
print(time_mean_with_time[2:] - time_mean_without_time)

for tp in tp_list:
    nc_with_time = nc4.Dataset(tp, 'r')
    nc_without_time = nc4.Dataset(without_time_nc4_data_path / tp.name, 'r')
    assert np.allclose(nc_with_time['meta'][:,2:], nc_without_time['meta'][:])