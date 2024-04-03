from git_ai.metrics.experiment import Experiment
from data_gen import DataGen

def main():
    data = DataGen()
    with Experiment() as e:
        writer = e.writer
        writer.add_hparams(metric_dict=data.before_metric_dict, hparam_dict={},
                           metric_unit_dict=data.before_metric_unit)
        for chk in data.checkpoints():
            for it in chk['iterations']:
                for plot in it:
                    writer.add_scalar(plot['name'], plot['value'], plot['step'])

            e.checkpoint(f"Checkpoint {chk['idx']}")

        writer.add_hparams(metric_dict=data.after_metric_dict, hparam_dict={},
                           metric_unit_dict=data.after_metric_dict)


if __name__ == '__main__':
    main()
