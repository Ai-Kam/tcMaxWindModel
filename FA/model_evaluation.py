import matplotlib.pyplot as plt
from tensorflow.keras import backend
import keras
import numpy as np
from pathlib import Path

from model_architecture import nc4PyDataset, MaxWindModel

use_time_info = False #True
use_without_time_info_data = False

max_wind_channel = 4 if not use_without_time_info_data else 2
meta_channnel = None if use_time_info or use_without_time_info_data else slice(2, None)

raw_data_path = Path('TC_data_GeoSciAI2024_test/')
models_path = Path('TC_data_GeoSciAI2024/') / 'concatenated_data_with_timeinfo'
nc4_data_path = raw_data_path / 'concatenated_data_with_timeinfo' if not use_without_time_info_data else raw_data_path / 'concatenated_data'
stats_path = nc4_data_path / 'stats'

tp_list = list(nc4_data_path.glob('*.nc'))


model_dirs = list(models_path.glob('models/*'))
target_run_num=52
run_name = next(mp.name for mp in model_dirs if mp.name.endswith(str(target_run_num)))
model_path = models_path / 'models' / run_name

stats = np.load(stats_path / 'time_stats.npz')
time_mean = stats['time_mean']
time_std = stats['time_std']

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square((y_pred - y_true)* time_std[max_wind_channel])))

#model = MaxWindModel()
#model.compile(optimizer='adam', loss=rmse)
model = keras.saving.load_model(model_path / "max_wind_model.keras", custom_objects={'MaxWindModel': MaxWindModel, 'rmse':rmse})
model.summary()

eval_dataset = nc4PyDataset(tp_list, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)

#eval_dataset.reset()
out_of_sample_loss = model.evaluate(eval_dataset)


y_pred = model.predict(eval_dataset)
eval_data = []
for i in range(len(eval_dataset)):
    eval_data.append((eval_dataset[i][1]))
y_true = np.concatenate(eval_data)
plt.scatter(y_true, y_pred, s=0.2)
coef = np.correlate(y_true, y_pred.flatten()) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred.flatten()))
x = np.linspace(min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred)), 100)
plt.plot(x, coef[0] * x, color='red')
plt.xlabel('True Values [MaxWind]')
plt.ylabel('Predictions [MaxWind]')
plt.axis('equal')
plt.axis('square')
plt.savefig(model_path / 'scatter.png')


plt.close()