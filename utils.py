import os
import random
import numpy as np

import torch
from sklearn import metrics

import json
import requests


def _lwlrap_sklearn(truth, scores):
    """Reference implementation from
    https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8"""
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.score = _lwlrap_sklearn(np.array(self.y_true), np.array(
            self.y_pred))
        return {
            "lwlrap": self.score
        }


def seed_everithing(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, content):
        with open(self.path, 'a') as appender:
            print(content)
            appender.write(content)


def notifyBayartsogt(message, slack_api_key, from_, platform="colab"):
    """
    ! curl -X POST -H 'Content-type: application/json' --data '{"text":"@Bayartsogt see this one!!"}' https://hook_url

    platform = "colab"
    from_ = "create_32k"
    message = "initing the notebook"
    notifyBayartsogt(platform, from_, message)
    """

    text = f"""
    | {platform} | {from_}
    |--- {message}
    """

    data = json.dumps({"text": text})
    # , headers={'Content-type: application/json'})
    return requests.post(slack_api_key, data=data).text
