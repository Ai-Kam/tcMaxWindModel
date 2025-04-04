import matplotlib.pyplot as plt
from tensorflow.keras import backend, callbacks
import numpy as np
from pathlib import Path
import wandb

from model_architecture import nc4PyDataset, MaxWindModel, MaxWindModel2

use_time_info = False #True
use_without_time_info_data = False

max_wind_channel = 4 if not use_without_time_info_data else 2
meta_channnel = None if use_time_info or use_without_time_info_data else slice(2, None)
# True False
# False False
# False True
max_epoch = 2

wandb.init(
    project='GeoSciAI2024',
    group='with_time_info' if (use_time_info and not use_without_time_info_data) else 'without_time_info',
    config={
    'model': 'MaxWindModel',
    'dataset': 'TC_data_GeoSciAI2024',
    'description': 'MaxWind prediction model',
    'epochs': max_epoch,
    },
    #name='01',
    )

raw_data_path = Path('TC_data_GeoSciAI2024/')
nc4_data_path = raw_data_path / 'concatenated_data_with_timeinfo' if not use_without_time_info_data else raw_data_path / 'concatenated_data'
stats_path = nc4_data_path / 'stats'
model_path = nc4_data_path / 'models' / wandb.run.name

if not (model_path).exists():
    (model_path).mkdir(parents=True)

attrs_meta = [
    'year',
    'month&date&hour',
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

tp_list = list(nc4_data_path.glob('*.nc'))
train = []
test = []
out_of_sample = []
for tp in tp_list:
    year = int(tp.stem[:4])
    if year >= 2002:
        out_of_sample.append(tp)
    elif year >= 2000:
        test.append(tp)
    else:
        train.append(tp)

stats = np.load(stats_path / 'time_stats.npz')
time_mean = stats['time_mean']
time_std = stats['time_std']
print(time_std)

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square((y_pred - y_true)* time_std[max_wind_channel])))

model = MaxWindModel()
#model.load_weights(model_path / 'max_wind_model_best.weights.h5')
model.compile(optimizer='adam', loss=rmse)
model.summary()

train_dataset = nc4PyDataset(train, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)
val_dataset = nc4PyDataset(test, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)
eval_dataset = nc4PyDataset(out_of_sample, max_wind_channnel=max_wind_channel, use_meta_channnel=meta_channnel)


class OutOfSampleEvaluationCallback(callbacks.Callback):
    def __init__(self, validation_data, wandb_run):
        super().__init__()
        self.validation_data = validation_data
        self.wandb_run = wandb_run

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self.model.evaluate(eval_dataset, verbose=0)
        
        self.wandb_run.log({
            'out_of_sample_loss': val_loss
        }, step=epoch)

early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=4, monitor='val_loss')
save_best = callbacks.ModelCheckpoint(model_path / 'max_wind_model_best.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
save_per_10epoches = callbacks.ModelCheckpoint(model_path / 'max_wind_model_{epoch:02d}.weights.h5', monitor='val_loss', save_weights_only=True, save_freq=10)
eval_callback = OutOfSampleEvaluationCallback(eval_dataset, wandb.run)
wandb_metrics_logger = wandb.keras.WandbMetricsLogger()

history = model.fit(
    x=train_dataset, 
    validation_data=val_dataset, 
    initial_epoch=0,
    epochs=max_epoch,
    callbacks=[early_stopping, reduce_lr, save_best, save_per_10epoches, eval_callback, wandb_metrics_logger]
)

history = history.history
wandb.log(history)

model.save_weights(model_path / 'max_wind_model_weights.weights.h5')
model.save(model_path / 'max_wind_model.keras')

eval_dataset.reset()
test_loss = model.evaluate(eval_dataset)
print(f'Test loss: {test_loss}')
wandb.log({'test_loss': test_loss})


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
plt.savefig(model_path / 'fig1.png')
wandb.log({'scatter': wandb.Image(str(model_path / 'fig1.png'))})

wandb.finish()

        