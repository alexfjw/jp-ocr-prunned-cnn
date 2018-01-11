from src.nn.models import *
from src.data.dataloaders import *
import argparse
import sys
import torch
import torch.backends.cudnn as cudnn
from src.prune import prune_model
from src.train import train_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--prune', dest='prune', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--train_path', type=str, default='train')
    parser.add_argument('--test_path', type=str, default='prune')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)

    return parser.parse_args()


def main():
    cudnn.benchmark = True
    args = get_args()

    data_loaders, num_classes = \
        get_etl2_dataloaders(args.model) if args.dataset == 'etl2' \
        else get_etl2_9g_dataloaders(args.model)

    if args.train:
        model, name = \
            vgg_model(num_classes) if args.model == "vgg" \
            else chinese_model(num_classes)
        model = train_model(model, data_loaders)
        torch.save(model.state_dict(), f'trained_models/{args.model}_{args.dataset}.weights')
    elif args.prune:
        model, name = vgg_model(num_classes) if args.model == "vgg" \
            else chinese_model(num_classes)
        model.load_state_dict(torch.load(f'trained_models/{args.model}_{args.dataset}.weights'))
        finetuning_passes = 10
        prune_model(model, data_loaders, finetuning_passes=finetuning_passes)
        torch.save(model.state_dict(),
                   f'trained_models/pruned_{args.model}_{args.dataset}_finetune{finetuning_passes}.weights')


if __name__ == '__main__':
    sys.exit(main())
