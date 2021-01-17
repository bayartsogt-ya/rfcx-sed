import os
import time
import numpy as np
import pandas as pd

import torch

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import AverageMeter, MetricMeter, seed_everithing, Logger, notifyBayartsogt
from datasets import SedDataset
from models import AudioSEDModel
from model_helpers import Mixup, do_mixup
from criterians import ImprovedPANNsLoss


# ----------------- Create Folds -----------------
def create_fold(data_dir, n_folds=5, seed=42):

    train = pd.read_csv(f"{data_dir}/train_tp.csv").sort_values("recording_id")

    train_gby = train.groupby("recording_id")[
        ["species_id"]].first().reset_index()
    train_gby = train_gby.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    train_gby.loc[:, 'kfold'] = -1

    X = train_gby["recording_id"].values
    y = train_gby["species_id"].values

    # split
    kfold = StratifiedKFold(n_splits=n_folds)
    MultilabelStratifiedKFold
    for fold, (t_idx, v_idx) in enumerate(kfold.split(X, y)):
        train_gby.loc[v_idx, "kfold"] = fold

    train = train.merge(
        train_gby[['recording_id', 'kfold']], on="recording_id", how="left")
    print(train.kfold.value_counts())
    train.to_csv("train_folds.csv", index=False)


def train_epoch(args, model, loader, criterion, optimizer,
                scheduler, mixup_augmenter, epoch):
    losses = AverageMeter()
    scores = MetricMeter()

    model.train()
    t = tqdm(loader)
    for i, sample in enumerate(t):
        optimizer.zero_grad()
        input = sample['image'].to(args.device)
        target = sample['target'].to(args.device)

        if mixup_augmenter:
            mixup_lambda = mixup_augmenter.get_lambda(
                batch_size=sample['image'].shape[0]).to(args.device)

            target = do_mixup(target, mixup_lambda)
        else:
            mixup_lambda = None

        output = model(input, mixup_lambda)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if scheduler and args.step_scheduler:
            scheduler.step()

        bs = input.size(0)
        scores.update(target, torch.sigmoid(
            torch.max(output['framewise_output'], dim=1)[0]))
        losses.update(loss.item(), bs)

        t.set_description(f"Train E:{epoch} - Loss{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg


def valid_epoch(args, model, loader, criterion, epoch):
    losses = AverageMeter()
    scores = MetricMeter()
    model.eval()
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample['image'].to(args.device)
            target = sample['target'].to(args.device)
            output = model(input)
            loss = criterion(output, target)

            bs = input.size(0)
            scores.update(target, torch.sigmoid(
                torch.max(output['framewise_output'], dim=1)[0]))
            losses.update(loss.item(), bs)
            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg


def test_epoch(args, model, loader):
    model.eval()
    pred_list = []
    id_list = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample["image"].to(args.device)
            bs, seq, w = input.shape
            input = input.reshape(bs * seq, w)
            id = sample["id"]
            output = model(input)
            output = torch.sigmoid(
                torch.max(output['framewise_output'], dim=1)[0])
            output = output.reshape(bs, seq, -1)
            # output = torch.sum(output, dim=1)
            output, _ = torch.max(output, dim=1)
            output = output.cpu().detach().numpy().tolist()
            pred_list.extend(output)
            id_list.extend(id)

    return pred_list, id_list


def train_fold(args):
    seed_everithing(args.seed)

    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    notifyBayartsogt(**args.slack_param)
    logger = Logger(f'{args.save_path}/log_fold_{args.fold}.txt')

    train_df = pd.read_csv(args.train_csv)
    sub_df = pd.read_csv(args.sub_csv)
    if args.DEBUG:
        train_df = train_df.sample(200)
    train_fold = train_df[train_df.kfold != args.fold]
    valid_fold = train_df[train_df.kfold == args.fold]

    train_dataset = SedDataset(
        df=train_fold,
        period=args.period,
        audio_transform=args.train_audio_transform,
        data_path=args.train_data_path,
        mode="train"
    )

    valid_dataset = SedDataset(
        df=valid_fold,
        period=args.period,
        stride=5,
        audio_transform=None,
        data_path=args.train_data_path,
        mode="valid"
    )

    test_dataset = SedDataset(
        df=sub_df,
        period=args.period,
        stride=5,
        audio_transform=None,
        data_path=args.test_data_path,
        mode="test"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    model = AudioSEDModel(**args.model_param)
    model = model.to(args.device)

    if args.pretrain_weights:
        print("---------------------loading pretrain weights")
        model.load_state_dict(torch.load(
            args.pretrain_weights, map_location=args.device), strict=False)
        model = model.to(args.device)

    # if args.is_train:
    # criterion = PANNsLoss()
    criterion = ImprovedPANNsLoss(**args.loss_param)
    # criterion = ImprovedFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # num_train_steps = int(len(train_loader) * args.epochs)
    # num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=num_train_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-8)

    # Augmenter

    if args.use_mixup:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    else:
        mixup_augmenter = None

    best_lwlrap = -np.inf
    early_stop_count = 0

    for epoch in range(args.start_epcoh, args.epochs):
        train_avg, train_loss = train_epoch(
            args, model, train_loader, criterion, optimizer,
            scheduler, mixup_augmenter, epoch)
        valid_avg, valid_loss = valid_epoch(
            args, model, valid_loader, criterion, epoch)

        if args.epoch_scheduler:
            scheduler.step()

        content = f"""
            {time.ctime()}
            Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}
            Train Loss:{train_loss:0.4f} - LWLRAP:{train_avg['lwlrap']:0.4f}
            Valid Loss:{valid_loss:0.4f} - LWLRAP:{valid_avg['lwlrap']:0.4f}
        """

        logger.log(content + '\n')
        if valid_avg['lwlrap'] > best_lwlrap:
            logger.log(
                f"########## >>>>>>>> Model Improved From {best_lwlrap} ----> {valid_avg['lwlrap']}")
            torch.save(model.state_dict(), os.path.join(
                args.save_path, f'fold-{args.fold}.bin'))
            best_lwlrap = valid_avg['lwlrap']
            early_stop_count = 0
        else:
            early_stop_count += 1

        if args.early_stop == early_stop_count:
            logger.log(
                f"\n $$$ ---? Ohoo.... we reached early stoping count : {early_stop_count}")
            logger.log(
                f"\n----------------\nFOLD {args.fold} | BEST_SCORE: {best_lwlrap}----------------\n")
            break

    model.load_state_dict(torch.load(os.path.join(
        args.save_path, f'fold-{args.fold}.bin'), map_location=args.device))
    model = model.to(args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = test_epoch(args, model, test_loader)
    print(np.array(test_pred).shape)

    test_pred_df = pd.DataFrame({
        "recording_id": sub_df.recording_id.values
    })
    test_pred_df[target_cols] = test_pred
    test_pred_df.to_csv(os.path.join(
        args.save_path, f"fold-{args.fold}-submission.csv"), index=False)

    logger.log(os.path.join(args.save_path,
                            f"fold-{args.fold}-submission.csv"))

    notifyBayartsogt(**args.slack_param)
