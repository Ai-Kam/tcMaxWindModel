import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

use_time_info = True

#raw_data_path = Path('TC_data_GeoSciAI2024/')
raw_data_path = Path('TC_data_GeoSciAI2024_test/')
nc4_data_path = raw_data_path / 'concatenated_data_with_timeinfo' if use_time_info else raw_data_path / 'concatenated_data'
stats_path = nc4_data_path / 'stats'

tp_list = list(raw_data_path.glob('track_data/*.csv'))


#exit()
attrs_meta = [
    'year',
    'month&date&hour',
    'central_lon',
    'central_lat',
    'max_wind',
    'central_pres',
    'stage'
] if use_time_info else [
    'central_lon',
    'central_lat',
    'max_wind',
    'central_pres',
    'stage'
]

attrs_fields = [
    'OLR',
    'QV600',
    'SLP',
    'SST',
    'U200',
    'U850',
    'V200',
    'V850'
]

meta_chan_num = len(attrs_meta)
fields_chan_num = len(attrs_fields)
total_chan_num = meta_chan_num + fields_chan_num

if not nc4_data_path.exists():
    nc4_data_path.mkdir()

if not stats_path.exists():
    stats_path.mkdir()

time_mean = np.zeros((total_chan_num,), dtype='float64')
squared_time_mean = np.zeros((total_chan_num,), dtype='float64')
n = 0

def load_data(year, attr, time, lon, lat):
    try:
        return np.load(raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon))%360):03}_{round(abs(lat)):03}{'s' if lat < 0 else 'n'}.npz")['data']
    except FileNotFoundError:
        #print(raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon))%360):03}_{round(abs(lat)):03}{'s' if lat < 0 else 'n'}.npz")
        if (raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon)+1)%360):03}_{round(abs(lat)):03}{'s' if lat < 0 else 'n'}.npz").exists():
            return np.load(raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon)+1)%360):03}_{round(abs(lat)):03}{'s' if lat < 0 else 'n'}.npz")['data']
        elif (raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon))%360):03}_{round(abs(lat)+1):03}{'s' if lat < 0 else 'n'}.npz").exists():
            return np.load(raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon))%360):03}_{round(abs(lat)+1):03}{'s' if lat < 0 else 'n'}.npz")['data']
        elif (raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon)+1)%360):03}_{round(abs(lat)+1):03}{'s' if lat < 0 else 'n'}.npz").exists():
            return np.load(raw_data_path / 'field_data' / str(year) / attr / f"{attr.lower()}_{time.strftime('%Y%m%d%H')}_{((round(lon)+1)%360):03}_{round(abs(lat)+1):03}{'s' if lat < 0 else 'n'}.npz")['data']

len_tp = len(tp_list)
for tp_count, tp in enumerate(tp_list):
    if (stats_path / f'time_stats_{(tp_count+10)//10*10}.npz').exists():
        n = np.load(stats_path  / f'time_stats_{(tp_count+10)//10*10}.npz')['n']
        time_mean = np.load(stats_path  / f'time_stats_{(tp_count+10)//10*10}.npz')['time_mean'] * n
        squared_time_mean = np.load(stats_path  / f'time_stats_{(tp_count+10)//10*10}.npz')['squared_time_mean'] * n
        continue
    track_data = open(tp, 'r').readlines()
    nc4_file = nc4.Dataset(nc4_data_path / f'{tp.stem}.nc', 'w', format='NETCDF4')
    nc4_file.createDimension('time', len(track_data)-1)
    nc4_file.createDimension('dummy_dim1', meta_chan_num)
    nc4_file.createDimension('dummy_dim2', fields_chan_num)
    nc4_file.createDimension('lon', 64)
    nc4_file.createDimension('lat', 64)
    nc4_file.createVariable('time', 'i8', ('time',))
    nc4_file.createVariable('metadata_names', 'S30', ('dummy_dim1',))
    nc4_file['metadata_names'][:] = np.array(list(attrs_meta), dtype='S10')
    nc4_file.createVariable('fields_names', 'S20', ('dummy_dim2',))
    nc4_file['fields_names'][:] = np.array(list(attrs_fields), dtype='S20')
    nc4_file.createVariable('lon', 'f4', ('lon',))
    nc4_file.createVariable('lat', 'f4', ('lat',))
    nc4_file.createVariable('meta', 'f4', ('time', 'dummy_dim1'))
    nc4_file.createVariable('fields', 'f4', ('time', 'dummy_dim2', 'lat', 'lon'))

    last_SST = None
    for t, snap in enumerate(track_data[1:]):
        snap = snap.replace('\n','').split(',')
        snap_time = datetime.strptime(snap[0], "%HZ%d%b%Y")
        nc4_file['time'][t] = int(round(snap_time.timestamp()))
        if use_time_info:
            time_stamp = datetime(snap_time.year, snap_time.month, snap_time.day, snap_time.hour) - datetime(snap_time.year, 1, 1, 0)
            nc4_file['meta'][t] = np.concatenate((np.array([int(snap_time.year), int(time_stamp.days) - int(time_stamp.seconds) / 86400], dtype=('float32')), np.array(snap[1:5], dtype='float32'), np.array(snap[6:7], dtype='float32')))
        else:
            nc4_file['meta'][t] = np.concatenate((np.array(snap[1:5], dtype='float32'), np.array(snap[6:7], dtype='float32')))
        time_mean[:meta_chan_num] += nc4_file['meta'][t]
        squared_time_mean[:meta_chan_num] += nc4_file['meta'][t] ** 2
        ns = 's' if float(snap[2]) < 0 else 'n'
        for i, attr in enumerate(attrs_fields):
            if attr != 'SST':
                tmp = load_data(snap_time.year, attr, snap_time, float(snap[1]), float(snap[2]))
                if np.any(np.isnan(tmp)):
                    tmp = np.nan_to_num(tmp, nan=0)
                if np.any(np.isnan(tmp)):
                    print(f'{attr} {snap_time} {float(snap[1])} {float(snap[2])} has nan')
                nc4_file['fields'][t,i] = tmp
                time_mean[i+meta_chan_num] += tmp.mean()
                squared_time_mean[i+meta_chan_num] += (tmp ** 2).mean()
            else:
                if snap_time.hour == 0:
                    tmp = load_data(snap_time.year, attr, snap_time, float(snap[1]), float(snap[2]))
                    if np.any(np.isnan(tmp)):
                        tmp = np.nan_to_num(tmp, nan=280)       # SST has a lot of nan (land), so fill it with 280
                    nc4_file['fields'][t,i] = tmp
                    time_mean[i+meta_chan_num] += tmp.mean()
                    squared_time_mean[i+meta_chan_num] += (tmp ** 2).mean()
                    if last_SST is None:
                        nc4_file['fields'][:t,i] = tmp
                        time_mean[i+meta_chan_num] += tmp.mean()*t
                        squared_time_mean[i+meta_chan_num] += (tmp ** 2).mean()*t
                    else:
                        nc4_file['fields'][t-3:t,i] = np.concatenate((last_SST*0.25+tmp*0.75, 0.5*last_SST + 0.5*tmp, last_SST*0.75+tmp*0.25), axis=0)
                        time_mean[i+meta_chan_num] += np.sum(np.mean(nc4_file['fields'][t-3:t,i], axis=(1,2)))
                        squared_time_mean[i+meta_chan_num] += np.sum(np.mean(nc4_file['fields'][t-3:t,i]**2, axis=(1,2)))
                    last_SST = tmp
        n += 1

    if snap_time.hour != 0:
        j = snap_time.hour // 6
        nc4_file['fields'][-j:, 3] = last_SST
        time_mean[meta_chan_num+3] += last_SST.mean()*j
        squared_time_mean[meta_chan_num+3] += (last_SST ** 2).mean()*j
    nc4_file.close()
    print(f'{tp.stem} done, {tp_count+1}/{len_tp}')

    if tp_count % 10 == 0:
        time_mean_tmp = time_mean / n
        squared_time_mean_tmp = squared_time_mean / n
        time_var_tmp = squared_time_mean_tmp - time_mean_tmp ** 2
        time_std_tmp = np.sqrt(time_var_tmp)
        np.savez(stats_path / f'time_stats_{tp_count}.npz', time_mean=time_mean_tmp, squared_time_mean=squared_time_mean_tmp, time_std=time_std_tmp, n=n)

time_mean /= n
squared_time_mean /= n
time_var = squared_time_mean - time_mean ** 2
time_std = np.sqrt(time_var)
np.savez(stats_path  / 'time_stats.npz', time_mean=time_mean, squared_time_mean=squared_time_mean, time_std=time_std, n=n)