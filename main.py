
import importlib
from argparse import ArgumentParser
from train_helpers import train_fold, create_fold

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])

    parser.add_argument('--args', type=str)
    params = parser.parse_args()

    module = importlib.import_module(params.args, package=None)
    args = module.args()

    create_fold(args.data_dir, n_folds=args.n_folds, seed=args.seed)
    train_fold(args=args)
