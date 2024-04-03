import base64
import math
import os
import random


class DataGen:
    METRIC_LABELS = [
        ('hparams/batch_size', '', 'FLOAT'), ('hparams/alpha', '', 'FLOAT'),
        ('hparms/beta', '', 'FLOAT'), ('hparams/dropout', '%', 'FLOAT'),
        ('hparams/optimizer', '', 'STRING'), ('hparams/shuffle', '', 'BOOLEAN'),
        ('hparams/lr', '', 'FLOAT'),
        ('f1_score', '', 'FLOAT'), ('training_loss', '', 'FLOAT'),
        ('test_loss', '', 'FLOAT'), ('mean_error', '%', 'FLOAT'),
        ('mean_squared_error', '%', 'FLOAT'), ('test_error', '%', 'FLOAT'),
        ('training_error', '%', 'FLOAT'), ('test_accuracy', '%', 'FLOAT'),
    ]

    PLOTS = [
        ('test_accuracy',  'FLOAT', 'log'),
        ('test_error', 'FLOAT', 'exponential'),
        ('test_loss', 'FLOAT', 'inverse_exponential'),
        ('training_accuracy', 'FLOAT', 'linear_growth'),
        ('training_error', 'FLOAT', 'linear_growth'),
    ]

    def __init__(self, checkpoints=10, iterations_per_chk=10) -> None:
        self.checkpoint_count = checkpoints
        self.iterations_per_chk = iterations_per_chk
        self.before_metric_dict = {
            label: self.__get_value(metric_type)
            for label, _, metric_type in self.METRIC_LABELS[:7]
        }
        self.before_metric_unit = {
            label: unit
            for label, unit, _ in self.METRIC_LABELS[:7]
        }

        self.after_metric_dict = {
            label: self.__get_value(metric_type)
            for label, _, metric_type in self.METRIC_LABELS[7:]
        }

        self.after_metric_unit = {
            label: unit
            for label, unit, _ in self.METRIC_LABELS[7:]
        }

    def checkpoints(self):
        return [
            {
                'idx': ckpt_idx,
                'iterations': [
                    self.iterations(ckpt_idx)
                ]
            }
            for ckpt_idx in range(self.checkpoint_count)
        ]

    def iterations(self, ckpt_idx):
        initial_it = ckpt_idx * self.iterations_per_chk
        return [
            {
                'name': label,
                'value': self.__sample_plot_value(initial_it + it_offset, plot_type, 10),
                'step': initial_it + it_offset
            }
            for it_offset, (label, _, plot_type) in enumerate(self.PLOTS)
        ]

    def __sample_plot_value(self, it, type, scale):
        if type == 'log':
            sampled_value = math.log((it+1)*0.01)*scale
        elif type == 'exponential':
            sampled_value = math.exp((it+1)*0.01)*scale
        elif type == 'inverse_exponential':
            sampled_value = 1/math.exp((it+1)*0.01)*scale
        elif type == 'linear_growth':
            sampled_value = it*scale
        elif type == 'linear_decrease':
            sampled_value = (self.total_iterations-it)*scale
        else:
            raise Exception('plot type unknown')

        noise = (random.normalvariate(.1*sampled_value, .01*sampled_value) -
                 .05*sampled_value)

        return sampled_value + noise

    def __get_value(self, metric_type):
        if metric_type == 'FLOAT':
            value = (random.random()*10)
        elif metric_type == 'INT':
            value = int(random.random()*1000)
        elif metric_type == 'STRING':
            value = random.choice(['ADAM', 'SGD', 'POWER_SGD'])
        elif metric_type == 'BOOLEAN':
            value = random.choice([True, False])
        else:
            raise Exception("metric type %s unknown" % metric_type)
        return value
