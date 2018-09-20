import random
import torch
import numpy as np
import argparse
import yaml
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from termcolor import colored

from segcore.loader import get_loader
from segcore.models import get_model
from segcore.optimizer import get_optimizer
from segcore.loss import get_loss

from segcore.loss.loss import *
from utils import *


def val(cfg=None, model=None):
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    v_loader = data_loader(cfg=cfg, training=False)
    val_loader = torch.utils.data.DataLoader(v_loader, batch_size=1)

    model.eval()
    all_preds = []
    all_gts = []

    for idx, (val_img_, val_label_, eroded_labels_) in enumerate(val_loader):

        val_img = np.squeeze(val_img_.numpy(), axis=(0,))
        val_label = np.squeeze(val_label_.numpy(), axis=(0,))
        eroded_labels = np.squeeze(eroded_labels_.numpy(), axis=(0,))

        pred = np.zeros(val_img.shape[:2] + (len(cfg["training"]["labels"]),))

        for i, coords in enumerate(grouper(cfg["training"]["batch_size"],
                                           sliding_window(val_img,
                                                          step=cfg["training"]["window_size"][0],
                                                          window_size=tuple(cfg["training"]["window_size"])))):

            # Build the tensor
            image_patches = [np.copy(val_img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            outs = model(image_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)
        all_preds.append(pred)
        all_gts.append(eroded_labels)

        # compute some metrics
        metrics(pred.ravel(), eroded_labels.ravel(), label_values=cfg["training"]["labels"])
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel(),
                           label_values=cfg["training"]["labels"])

        return accuracy


def train(cfg=None):

    #basic parameters
    train_para = cfg["training"]
    epochs = train_para["epochs"]
    batch_size = train_para["batch_size"]
    device_ids = train_para["device_ids"]
    labels = train_para["labels"]
    n_classes = len(labels)
    seeds = train_para["seeds"]

    # Setup seeds
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    np.random.seed(seeds)
    random.seed(seeds)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    t_loader = data_loader(cfg=cfg)
    train_loader = torch.utils.data.DataLoader(t_loader, batch_size=batch_size)

    # set the model
    model = get_model(cfg["model"]["arch"], n_classes=n_classes)
    model = torch.nn.DataParallel(model(), device_ids=device_ids).cuda()

    # set the optimizer and scheduler, loss function
    optimizer_type = get_optimizer(cfg=cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() if k != 'name'}
    optimizer = optimizer_type(model.parameters(), **optimizer_params)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
    losser = get_loss(cfg=cfg)

    # to iter
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)

    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data.cuda()), Variable(target.cuda())

            optimizer.zero_grad()
            output = model(data)

            loss = losser(input=output, target=target)
            loss.backward()

            optimizer.step()

            losses[iter_] = loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')

                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print(colored('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0], accuracy(pred, gt)), 'red', 'on_yellow'))


            iter_ += 1

            del (data, target, loss)

        if e % 1 == 0:
            # We validate with the largest possible stride for faster computing
            acc = val(cfg=cfg, model=model)
            torch.save(model.state_dict(), './model_paras/segnet256_epoch{}_{}'.format(e, acc))
    torch.save(model.state_dict(), './model_paras/segnet_final')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/isprs_linknet.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    train(cfg=cfg)