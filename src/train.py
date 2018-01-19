from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


def train_model(model, dataloaders):
    print('training model:', model)

    # Checks if GPU is available
    use_gpu = torch.cuda.is_available()
    since = time.time()

    # Keeps track of best parameters and f1 score from validation phase
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 25

    print()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Train and validate for each epoch
        for phase in ['train', 'val']:
            dataloader = dataloaders[phase]

            if phase == 'train':
                dataloader.dataset.train = True
                model.train(True)
                scheduler.step()
            else:
                dataloader.dataset.train = False
                model.train(False)


            # Keeps track of epoch loss, labels vs predictions
            running_loss = 0.0

            running_labels = torch.LongTensor()
            running_predictions = torch.LongTensor()

            if use_gpu:
                model = model.cuda()

            # Iterate data using dataloaders
            for data in tqdm.tqdm(dataloader):
                inputs, labels = data

                # Wrap inputs and labels in Variables
                if use_gpu:
                    is_val = phase == 'val'
                    inputs = Variable(inputs.cuda(), volatile=is_val)
                    labels = Variable(labels.cuda(), volatile=is_val)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward and loss calculation
                outputs = model(inputs)
                _, pred_indices = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Backward and optimize if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_labels = torch.cat((running_labels, labels.data.cpu()), 0)
                running_predictions = torch.cat((running_predictions, pred_indices.cpu()), 0)

                del inputs, labels

            loader_size = len(dataloader.batch_sampler.sampler)
            epoch_loss = running_loss / loader_size

            # Calculate f1_score using true labels and predictions
            epoch_f1 = f1_score(running_labels.numpy(), running_predictions.numpy(), average='micro')

            print('{} Loss: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_f1))

            # Update best parameters and f1 score if in validation phase
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = model.state_dict()

        epoch_elapsed = time.time() - epoch_start
        print('Epoch {} took in {:.0f}m {:.0f}s'.format(
            epoch, epoch_elapsed // 60, epoch_elapsed % 60))
        print()

    # Compute total time
    since = time.time()
    time_elasped = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elasped // 60, time_elasped % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # load model with optimized parameters
    model.load_state_dict(best_model_wts)

