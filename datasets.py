import math
import numpy as np
import soundfile as sf


def crop_or_pad(y, sr, period, record, mode="train"):
    len_y = len(y)
    effective_length = sr * period
    rint = np.random.randint(len(record['t_min']))
    time_start = record['t_min'][rint] * sr
    time_end = record['t_max'][rint] * sr
    if len_y > effective_length:
        # Positioning sound slice
        center = np.round((time_start + time_end) / 2)
        beginning = center - effective_length / 2
        if beginning < 0:
            beginning = 0
        beginning = np.random.randint(beginning, center)
        ending = beginning + effective_length
        if ending > len_y:
            ending = len_y
        beginning = ending - effective_length
        y = y[beginning:ending].astype(np.float32)
    else:
        y = y.astype(np.float32)
        beginning = 0
        ending = effective_length

    beginning_time = beginning / sr
    ending_time = ending / sr
    label = np.zeros(24, dtype='f')

    for i in range(len(record['t_min'])):
        if (record['t_min'][i] <= ending_time) and (record['t_max'][i] >= beginning_time):
            label[record['species_id'][i]] = 1

    return y, label


class SedDataset:
    def __init__(self, df, period=10, stride=5,
                 audio_transform=None, data_path="train", mode="train"):

        self.period = period
        self.stride = stride
        self.audio_transform = audio_transform
        self.data_path = data_path
        self.mode = mode

        self.df = df.groupby("recording_id").agg(
            lambda x: list(x)).reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]

        y, sr = sf.read(f"{self.data_path}/{record['recording_id']}.flac")

        if self.mode != "test":
            y, label = crop_or_pad(
                y, sr, period=self.period, record=record, mode=self.mode)

            if self.audio_transform:
                y = self.audio_transform(samples=y, sample_rate=sr)
        else:
            i = 0
            effective_length = self.period * sr
            # stride = self.stride * sr

            # y = np.stack([y[i: i + effective_length].astype(np.float32)
            #               for i in range(0, 60 * sr + stride - effective_length, stride)])

            y = np.stack([y[i * effective_length: (i + 1) * effective_length].astype(np.float32)
                          for i in range(0, math.ceil(60 / self.period))])

            label = np.zeros(24, dtype='f')
            if self.mode == "valid":
                for i in record['species_id']:
                    label[i] = 1

        return {
            "image": y,
            "target": label,
            "id": record['recording_id']
        }
