import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.pytorch_modelsize import SizeEstimator
from sklearn.metrics import f1_score
from datetime import datetime
from tqdm import tqdm


def model_to_weights(source, dest):
    x = torch.load(source)
    torch.save(x.state_dict(), dest)


def benchmark(model, val_dataloader, header):
    # get estimated memory consumption
    se = SizeEstimator(model)
    se.get_parameter_sizes()
    se.calc_param_bits()
    bytes = se.param_bits // 8
    mbytes = bytes / 1e6

    print(f'Benchmark, {header}: ')
    print(f'Size of model: {mbytes} MB')

    # show accuracy using validation set

    model.train(False)
    val_dataloader.dataset.train = False

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_labels = torch.LongTensor()
    running_predictions = torch.LongTensor()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    for data in tqdm(val_dataloader):
        inputs, labels = data
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        # Forward and loss calculation
        outputs = model(inputs)
        _, pred_indices = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]
        running_labels = torch.cat((running_labels, labels.data.cpu()), 0)
        running_predictions = torch.cat((running_predictions, pred_indices.cpu()), 0)

        del inputs, labels

    loader_size = len(val_dataloader.batch_sampler.sampler)
    epoch_loss = running_loss / loader_size

    # Calculate f1_score using true labels and predictions
    epoch_f1 = f1_score(running_labels.numpy(), running_predictions.numpy(), average='macro')

    print('Loss: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_f1))

    # show time taken for single input
    model = model.cpu()
    inputs, _ = next(iter(val_dataloader))
    t0 = datetime.now()
    model(Variable(inputs, volatile=True))
    t1 = datetime.now()
    print(f'Time taken for inference: {(t1-t0).total_seconds()/val_dataloader.batch_size}s')


