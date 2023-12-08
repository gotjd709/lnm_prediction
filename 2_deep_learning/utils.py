from segmentation_models_pytorch.utils.meter    import AverageValueMeter
from datetime                                   import date
from tqdm                                       import tqdm
from configs                                    import *
import matplotlib.pyplot                        as plt
import numpy                                    as np
import torch
import sys
import os

### Global Variables from config

class EarlyStopping:
    '''
    stop training if the validation loss dose not imporved within the patience
    '''
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, scheduler, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


def name_path():
    '''
    name the weight path and tensorboard path
    '''
    mode = 'binary_class' if CLASSES==1 else 'multi_class'
    name = f'{mode}_{MODEL}_{ENCODER_NAME}_{LOSS}_{DESCRIPTION}_{date.today()}'
    os.makedirs(LOG_PATH, exist_ok = True)
    os.makedirs(WEIGHT_PATH, exist_ok = True)
    return f'{WEIGHT_PATH}/{name}.pth', f'{LOG_PATH}/{name}'


def viz_img_msk(BATCH_SIZE, idx, x, y, yp):
    '''
    visualize images, ground truths and prediction masks
    '''    
    npx1 = x.numpy()
    y = y.cpu().detach().numpy().round()
    y = np.argmax(y, axis=0)
    yp = yp.cpu().detach().numpy().round()
    yp = np.argmax(yp, axis=0)
    plt.subplot(BATCH_SIZE,3,idx*3+1)
    plt.imshow(np.transpose(npx1, (1, 2, 0)))
    plt.axis('off')
    plt.title('patch image')
    plt.subplot(BATCH_SIZE,3,idx*3+2)
    plt.imshow(y, vmin=0, vmax=3, cmap='Oranges')
    plt.axis('off')
    plt.title('ground truth')
    plt.subplot(BATCH_SIZE,3,idx*3+3)
    plt.imshow(yp, vmin=0, vmax=3, cmap='Oranges')
    plt.axis('off')
    plt.title('prediction')
    # plt.savefig(f'/workspace/data/figure_zip/1_deep_learning/10_patch_level_results/101_external_datasets/{idx}.png', dpi=DPI)
    plt.cla()


def check_predictions(BATCH_SIZE, device, model, xs, ys):
    '''
    visualize batch products into tensorboard
    '''
    yps = model(xs.to(device))
    ys = ys.to(device)
    fig = plt.figure(figsize=(4*3, 4*(BATCH_SIZE+1)))
    for idx, (x, y, yp) in enumerate(zip(xs, ys, yps)):
        viz_img_msk(BATCH_SIZE, idx, x, y, yp)
    return fig

