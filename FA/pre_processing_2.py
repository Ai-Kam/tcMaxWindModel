import netCDF4 as nc4
import numpy as np
from pathlib import Path

use_time_info = True

#raw_data_path = Path('TC_data_GeoSciAI2024/')
raw_data_path = Path('TC_data_GeoSciAI2024_test/')
nc4_data_path = raw_data_path / 'concatenated_data_with_timeinfo' if use_time_info else raw_data_path / 'concatenated_data'
stats_path = nc4_data_path / 'stats'

tp_list = list(nc4_data_path.glob('*.nc'))
stats = np.load(stats_path / 'time_stats.npz')
time_mean = stats['time_mean']
time_std = stats['time_std']
meta_chan_num = 7 if use_time_info else 5

for tp in tp_list:
    nc = nc4.Dataset(tp, 'r+')
    if hasattr(nc, 'status'):
        if nc.status == 'normalized':
            continue
    nc['meta'][:] -= time_mean[:meta_chan_num]
    nc['meta'][:] /= time_std[:meta_chan_num]
    nc['fields'][:] -= time_mean[meta_chan_num:].reshape(1, -1, 1, 1)
    nc['fields'][:] /= time_std[meta_chan_num:].reshape(1, -1, 1, 1)
    nc.setncattr('status', 'normalized')
    nc.close()
    print(f'Normalized {tp.name}')