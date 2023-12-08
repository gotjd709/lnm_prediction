from torch.utils.tensorboard                                 import SummaryWriter
from torchsampler                                            import ImbalancedDatasetSampler
from utils                                                   import EarlyStopping, TrainEpoch, ValidEpoch, name_path, check_predictions
from datagen                                                 import TensorData
from sklearn.utils                                           import shuffle
from configs                                                 import *
import segmentation_models_pytorch                           as smp
import torch.optim                                           as optim
import torch.nn                                              as nn
import torch 
import pickle

### Global Variables from config

def model_setting():
    model = getattr(smp, MODEL)(
        encoder_name    = ENCODER_NAME, 
        encoder_weights = ENCODER_WEIGHTS, 
        classes         = CLASSES, 
        activation      = ACTIVATION
    )
    model = model.to(DEVICE)
    model = nn.DataParallel(model, device_ids=[0,1])
    return model


def dataloader_setting():
    # Path Loading
    with open(TRAIN_PATH, 'rb') as fr:
        TRAIN_ZIP = pickle.load(fr)
    with open(VALID_PATH, 'rb') as fr:
        VALID_ZIP = pickle.load(fr)
    with open(TEST_PATH, 'rb') as fr:
        TEST_ZIP = pickle.load(fr)
    TEST_ZIP = shuffle(TEST_ZIP, random_state=RANDOM_SEED)
    external_patch_zip  = [(x.replace('input_y100','input_x100'), x) for x in EXTERNAL_PATCH_LIST]
    TEST_ZIP.extend(external_patch_zip)

    # Dataset Setting
    train_data = TensorData(TRAIN_ZIP, INPUT_SHAPE, CLASSES, augmentation=True)
    valid_data = TensorData(VALID_ZIP, INPUT_SHAPE, CLASSES)
    test_data = TensorData(TEST_ZIP, INPUT_SHAPE, CLASSES)
    
    # DataLoader Setting
    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        sampler = ImbalancedDatasetSampler(train_data) if SAMPLER else None,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_data,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_data,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    return train_loader, valid_loader, test_loader


def train_dataset():
    '''
    train deep learning model
    '''
    # model setting
    model = model_setting()
    
    # dataloader setting
    train_loader, valid_loader, test_loader = dataloader_setting()

    # weight and log setting
    weight, log_dir = name_path()
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=DESCRIPTION)

    # loss, metrics, optimizer and schduler setting
    loss = getattr(smp.utils.losses, LOSS)()
    metrics = [smp.utils.metrics.IoU(), smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy(), smp.utils.metrics.Recall(), smp.utils.metrics.Precision()]
    metrics = [smp.utils.metrics.IoU(ignore_channels=[0,1,3]), smp.utils.metrics.Fscore(ignore_channels=[0,1,3]), smp.utils.metrics.Accuracy(ignore_channels=[0,1,3]), smp.utils.metrics.Recall(ignore_channels=[0,1,3]), smp.utils.metrics.Precision(ignore_channels=[0,1,3])]
    metrics = [smp.utils.metrics.IoU(ignore_channels=[0,1,2]), smp.utils.metrics.Fscore(ignore_channels=[0,1,2]), smp.utils.metrics.Accuracy(ignore_channels=[0,1,2]), smp.utils.metrics.Recall(ignore_channels=[0,1,2]), smp.utils.metrics.Precision(ignore_channels=[0,1,2])]
    optimizer = getattr(optim, OPTIMIZER)(params=model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    patience = PATIENCE
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=weight)

    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    for i in range(0, EPOCH):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        writer.add_scalars('Loss', {'train_loss':train_logs['loss'],
                            'valid_loss':valid_logs['loss']}, i)
        writer.add_scalars('IoU', {'train_loss':train_logs['iou_score'],
                                    'valid_loss':valid_logs['iou_score']}, i)
        writer.add_scalars('Fscore', {'train_loss':train_logs['fscore'],
                                    'valid_loss':valid_logs['fscore']}, i)

        xs, ys = next(iter(valid_loader))
        writer.add_figure('check predictions !',
                        check_predictions(BATCH_SIZE, DEVICE, model, xs, ys),
                        global_step=i)

        early_stopping(valid_logs['loss'], model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    model.load_state_dict(torch.load(weight))
    test_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    test_logs = test_epoch.run(test_loader)
    writer.add_scalars('Loss', {'test':test_logs['loss']}, i+1)
    writer.add_scalars('IoU', {'test':test_logs['iou_score']}, i+1)
    writer.add_scalars('Fscore', {'test':test_logs['fscore']}, i+1)
    xs, ys = next(iter(test_loader))
    check_predictions(BATCH_SIZE, DEVICE, model, xs, ys)
    writer.add_figure('check test datasets !', check_predictions(BATCH_SIZE, DEVICE, model, xs, ys))    



if __name__ == '__main__':
    train_dataset()