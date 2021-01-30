import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from apex import amp
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from gen_data import MyDataset
from model import MyModel
from MyLoss import LabelSmoothingCrossEntropy,FocalLoss


class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emd_name="emb"):
        for name, params in self.model.named_parameters():
            if params.requires_grad is True and emd_name in name:
                self.backup[name] = params.data.clone()
                norm = torch.norm(params.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * params.grad / norm
                    params.grad.add_(r_at)

    def restore(self, emd_name="emb"):
        for name, params in self.model.named_parameters():
            if params.requires_grad is True and emd_name in name:
                assert name in self.backup
                params.data = self.backup[name]

        self.backup = {}

class Trainer(object):
    def __init__(self, model, fgm, accum_num, fold_num, train_loader, dev_loader, load_save, loss_fn, optimizer, \
                 scheduler, save_path, epochs, writer, max_norm, eval_step_interval, best_f1, device):

        super(Trainer, self).__init__()

        self.model = model
        self.fgm = fgm
        self.accum_num = accum_num
        self.fold_num = fold_num
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.load_save = load_save
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.epochs = epochs
        self.writer = writer
        self.max_norm = max_norm
        self.eval_step_interval = eval_step_interval
        self.device = device
        self.best_f1 = best_f1

        if self.load_save is True:
            self.model.load_state_dict(torch.load(self.save_path)["model"], strict=False)
            print("Sucessfully load model!")
            print("initial best F1 is:{best_f1}".format(best_f1=self.best_f1))

        self.model.to(self.device)

        # amp
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # amp

    def train(self):
        self.model.train()
        global_step = 1
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs + 1):
            for idx, batch_data in enumerate(self.train_loader, start=1):
                image_data, input_ids, attention_mask, label = batch_data[0], batch_data[1], \
                                                               batch_data[2], batch_data[3]

                logits = self.model(image_data.to(self.device), input_ids.to(self.device), \
                                    attention_mask.to(self.device))
                loss = self.loss_fn(logits, label.to(self.device))
                #loss = loss / self.accum_num
                self.writer.add_scalar("%d_train/loss" % self.fold_num, loss.item(), global_step=global_step)

                # --------amp----------------
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # --------amp----------------

                # 对抗训练
                fgm = self.fgm(self.model)
                fgm.attack()
                logits_adv = self.model(image_data.to(self.device), input_ids.to(self.device), \
                                        attention_mask.to(self.device))
                loss_adv = self.loss_fn(logits_adv, label.to(self.device))

                loss_adv = loss_adv / self.accum_num

                # ---------------amp-----------------
                with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss_adv:
                    scaled_loss_adv.backward()
                # ---------------amp------------------

                # loss_adv.backward()
                fgm.restore()
                # 对抗训练

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

                if idx % self.accum_num == 0:
                    print(datetime.now(), "---",
                          "epoch:{epoch},step:{step},train_loss:{loss}.".format(epoch=epoch, step=idx, \
                                                                                loss=loss.item()))
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    #self.ema.update()

                global_step += 1

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            #self.ema.update()

            p, r, f1 = self.eval()
            self.model.train()
            self.writer.add_scalar("%d_val/p" % self.fold_num, p, global_step)
            self.writer.add_scalar("%d_val/r" % self.fold_num, r, global_step)
            self.writer.add_scalar("%d_val/f1" % self.fold_num, f1, global_step)

            if self.best_f1 < f1:
                self.best_f1 = f1

                ckpt_dict = {
                    "model": self.model.state_dict()
                }
                torch.save(ckpt_dict, f=self.save_path)

            print(datetime.now(), "---", \
                  "epoch:{epoch},precision:{p},recall:{r},F1-score:{f1},{fold_num}_best_F1:{best_f1}".format(
                      epoch=epoch, p=p, r=r, f1=f1, fold_num=self.fold_num,best_f1=self.best_f1))
            print("------end evaluating model in dev data------")

        self.writer.flush()
        self.writer.close()

    def eval(self):
        self.model.eval()
        #self.ema.apply_shadow()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for idx, batch_data in enumerate(self.dev_loader, start=1):
                image_data, input_ids, attention_mask, label = batch_data[0], batch_data[1], \
                                                               batch_data[2], batch_data[3]
                logits = self.model(image_data.to(self.device), input_ids.to(self.device), \
                                    attention_mask.to(self.device))
                y_true.extend(label)

                logits = logits.cpu().numpy()
                for item in logits:
                    y_pred.append(np.argmax(item))

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        p = precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
        r = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")

        cls_report = classification_report(y_true=y_true, y_pred=y_pred)
        print("------start evaluating model in dev data------")
        print(datetime.now(), "---")
        print(cls_report)

        #self.ema.restore()

        return p, r, f1



def K_fold_training(model_name, data_np, fold_num, label2idx, tokenizer, \
                    resnet_path, bert_path, dropout_rate, num_class, \
                    epochs, batch_size, accum_num, base_save_path, max_norm, \
                    eval_step_interval, base_writer_path, device):
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True)
    nfold = 1

    # k折训练
    for train_idx, val_idx in skf.split(data_np[:, :2], data_np[:, 2:]):
        # 训练集、验证集的idx
        train_data = data_np[train_idx, :]
        dev_data = data_np[val_idx, :]

        # 训练集、验证集
        train_data = MyDataset(train_data, label2idx, tokenizer)
        dev_data = MyDataset(dev_data, label2idx, tokenizer)

        # train_loader、dev_loader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, drop_last=True)

        # 定义模型、损失函数、优化器
        model = MyModel(model_name=model_name, resnet_path=resnet_path, bert_path=bert_path, \
                        dropout_rate=dropout_rate, num_class=num_class)

        loss_fn = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=(epochs // 4) + 1, eta_min=2e-8)

        # 实例化trainer
        save_path = base_save_path + "/%d_best.ckpt" % nfold
        writer = SummaryWriter(base_writer_path + "/%d" % nfold)
        trainer = Trainer(model=model, fold_num=nfold, fgm=FGM, accum_num=accum_num, train_loader=train_loader, \
                          dev_loader=dev_loader, load_save=False, loss_fn=loss_fn, optimizer=optimizer, \
                          scheduler=scheduler, save_path=save_path, epochs=epochs, writer=writer, max_norm=max_norm, \
                          eval_step_interval=eval_step_interval, best_f1=0.0, device=device)
        trainer.train()
        print("{id} fold best f1 is :{best_f1}".format(id=nfold, best_f1=trainer.best_f1))

        model=model.cpu()
        torch.cuda.empty_cache()

        del model
        del train_data
        del train_loader
        del dev_data
        del dev_loader
        del optimizer
        del loss_fn
        del trainer

        nfold += 1

    print("finished all training! lucky!")
