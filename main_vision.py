import torch
import time
import numpy as np

import logger
from arg_parser import arg_parser
from data import get_dataloader
from models import FullModel
import model_utils


def load_model_and_optimizer(opt, num_GPU=None, reload_model=False):
    model = FullModel.FullVisionModel(opt)
    optimizer = []
    for idx, layer in enumerate(model.encoder[:3]):
        optimizer.append(torch.optim.Adam(layer.parameters(), lr=opt.learning_rate))

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    model, optimizer = model_utils.reload_weights(
        opt, model, optimizer, reload_model=reload_model
    )

    return model, optimizer


def train(opt, model):
    total_step = len(train_loader)

    starttime = time.time()
    cur_train_module = 3
    print_idx = 100

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        loss_epoch = [0, 0, 0]
        loss_updates = [1, 1, 1]

        for step, (img, label) in enumerate(train_loader):

            img = img.to(opt.device)

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            loss, _ = model(img, n=cur_train_module)
            loss = torch.mean(loss, 0)

            for idx, cur_losses in enumerate(loss):
                model.zero_grad()

                if idx == len(loss) - 1:
                    cur_losses.backward()
                else:
                    cur_losses.backward(retain_graph=True)
                optimizer[idx].step()

                print_loss = cur_losses.item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer)



if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_model_and_optimizer(opt)
    logs = logger.Logger(opt)
    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)
    train(opt, model)
    logs.create_log(model)
