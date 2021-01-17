
import importlib
from argparse import ArgumentParser
from train_helpers import train_fold, create_fold

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])

    parser.add_argument('--args', type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--message', type=str)
    parser.add_argument('--slack_api_key', type=str)
    parser.add_argument('--from_', type=str)
    params = parser.parse_args()

    module = importlib.import_module(params.args, package=None)
    args = module.args()

    args.slack_param = {
        "message": params.message,
        "slack_api_key": params.slack_api_key,
        "from_": params.from_,
    }
    args.fold = params.fold

    create_fold(args.data_dir, n_folds=args.n_folds, seed=args.seed)

    print("-" * 20)
    print("FOLD", args.fold)
    print("-" * 20)

    train_fold(args=args)
