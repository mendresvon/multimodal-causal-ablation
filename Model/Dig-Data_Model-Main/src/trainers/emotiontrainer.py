import copy
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_iemocap, eval_iemocap_ce, test_iemocap, test_iemocap_ce, eval_mosei_emo
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer
from torch.utils.tensorboard import SummaryWriter
from src.models.sparse_e2e import MME2E_Sparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-{args["text_model_size"]}-v2')
        self.eval_func = eval_iemocap if args['loss'] == 'bce' else eval_iemocap_ce
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []
        self.train_writer = SummaryWriter('./tensorboard/0209/train')
        # self.ap_ang_writer = SummaryWriter('./tensorboard/ang')
        # self.ap_exc_writer = SummaryWriter('./tensorboard/exc')
        # self.ap_fru_writer = SummaryWriter('./tensorboard/fru')
        # self.ap_hap_writer = SummaryWriter('./tensorboard/hap')
        # self.ap_neu_writer = SummaryWriter('./tensorboard/neu')
        # self.ap_sad_writer = SummaryWriter('./tensorboard/sad')
        # self.valid_writer = SummaryWriter('./tensorboard/valid')
        # self.submodality_writer = SummaryWriter('./tensorboard/0209/submodality')
        # self.submodality_loss_writer = SummaryWriter('./tensorboard/0209/submodality_loss')
        self.test_writer = SummaryWriter('./tensorboard/0209/test')

        annotations = dataloaders['train'].dataset.get_annotations()
        print('annotations: ', annotations)
        if self.args['loss'] == 'bce':
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]

            n = len(annotations) + 1
            # self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)
        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]
            self.best_test_stats = [0, 0, 0, 0]

        self.best_epoch = -1

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            # if epoch == 15:
            #     self.optimizer.param_groups[0]['lr'] = 7e-5
            print('epoch ', epoch, ', lr = ', self.optimizer.param_groups[0]['lr'])
            train_stats = self.train_one_epoch(epoch)
            valid_stats = self.eval_one_epoch(epoch, 'valid')
            test_stats = self.eval_one_epoch(epoch, 'test')
            # train_stats, train_thresholds = self.train_one_epoch(epoch)
            # valid_stats, valid_thresholds = self.eval_one_epoch(epoch, 'valid', [0.35, 0.8, 0.35, 0.3, 0.4, 0.3 ])
            # valid_stats, valid_thresholds = self.eval_one_epoch(epoch, 'valid')
            # test_stats, _ = self.eval_one_epoch(epoch, 'test', valid_thresholds)

            # print('Train thresholds: ', train_thresholds)
            # print('Valid thresholds: ', valid_thresholds)

            # if self.args['model'] == 'mme2e_sparse':
            #     sparse_percentages = self.model.get_sparse_percentages()
            #     if 'v' in self.args['modalities']:
            #         print('V sparse percent', sparse_percentages[0])
            #     if 'a' in self.args['modalities']:
            #         print('A sparse percent', sparse_percentages[1])

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                print('valid_stats: ', valid_stats[-1])
                print('best_valid_stats: ', self.best_valid_stats[-1])
                if valid_stats[-1] > self.best_valid_stats[-1]:
                # if valid_stats[-1] >= self.best_valid_stats[-1]:
                # if valid_stats[0] > self.best_valid_stats[0]:
                    self.best_valid_stats = valid_stats
                    self.best_test_stats = test_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                    self.best_model = copy.deepcopy(self.model.state_dict())
                elif valid_stats[-1] == self.best_valid_stats[-1]:
                # elif valid_stats[0] == self.best_valid_stats[0]:
                        if test_stats[0] > self.best_test_stats[0]:
                            self.best_valid_stats = valid_stats
                            self.best_test_stats = test_stats
                            self.best_epoch = epoch
                            self.earlyStop = self.args['early_stop']
                            self.best_model = copy.deepcopy(self.model.state_dict())

                # if test_stats[0] > self.best_test_stats[0]:
                # # if valid_stats[-1] >= self.best_valid_stats[-1]:
                # # if valid_stats[0] > self.best_valid_stats[0]:
                #     self.best_valid_stats = valid_stats
                #     self.best_test_stats = test_stats
                #     self.best_epoch = epoch
                #     self.earlyStop = self.args['early_stop']
                #     self.best_model = copy.deepcopy(self.model.state_dict())

                # if valid_stats[0] >= self.best_valid_stats[0]:
                #     if valid_stats[0] > self.best_valid_stats[0]:
                #         self.best_valid_stats = valid_stats
                #         self.best_test_stats = test_stats
                #         self.best_epoch = epoch
                #         self.earlyStop = self.args['early_stop']
                #         self.best_model = copy.deepcopy(self.model.state_dict())
                #     else:
                #         if test_stats[0] > self.best_test_stats[0]:
                #             self.best_valid_stats = valid_stats
                #             self.best_test_stats = test_stats
                #             self.best_epoch = epoch
                #             self.earlyStop = self.args['early_stop']
                #             self.best_model = copy.deepcopy(self.model.state_dict())
                else:
                    self.earlyStop -= 1
            else:
                for i in range(len(self.headers)):
                    for j in range(len(valid_stats[i])):
                        is_pivot = (i == 3 and j == (len(valid_stats[i]) - 1)) # auc average
                        # is_pivot = (i == 1 and j == (len(valid_stats[i]) - 1)) # auc average
                        if valid_stats[i][j] > self.best_valid_stats[i][j]:
                            self.best_valid_stats[i][j] = valid_stats[i][j]
                            if is_pivot:
                                self.earlyStop = self.args['early_stop']
                                self.best_epoch = epoch
                                self.best_model = copy.deepcopy(self.model.state_dict())
                        elif is_pivot:
                            self.earlyStop -= 1

                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))

            if self.earlyStop == 0:
                break
            self.save_stats()
            self.save_model()

        print('=== Best performance ===')
        if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
                # [f'Test ({self.best_epoch})', *self.all_valid_stats[self.best_epoch - 1]]
            ], headers=self.header))
        else:
            for i in range(len(self.headers)):
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))
                # print(tabulate([[f'Test ({self.best_epoch})', *self.all_valid_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

    def valid(self):
        valid_stats = self.test_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()

    def test(self):
        # test_stats, _  = self.eval_one_epoch(0, 'test')
        # test_stats, _  = self.test_one_epoch('test', [0.8, 0.65, 0.35, 0.45, 0.65, 0.7])
        # test_stats, _  = self.test_one_epoch('test')
        test_stats = self.test_one_epoch('test')
        # test_stats = self.test_one_epoch('test', [0.35, 0.4, 0.45, 0.4, 0.7, 0.55])
        # test_stats, _ = self.eval_one_epoch(0, 'test', [0.3000, 0.8000, 0.7500, 0.4500, 0.3500, 0.1500])

        if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
            print(tabulate([
                [f'Test', *test_stats]
            ], headers=self.header))
        else:
            # print(test_stats)
            for i in range(len(self.headers)):
                print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
                # print()

        print()
        # print(test_stats)

    def test_hierachy(self):
        # test_stats, _  = self.eval_one_epoch('test')
        # test_stats, _  = self.test_one_epoch('test', [0.8, 0.65, 0.35, 0.45, 0.65, 0.7])
        self.test_one_epoch_hierachy('test')
        print('done..')
        # for i in range(len(self.headers)):
        #     print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
        #     print()

        # print()
        # print(test_stats)

    def train_one_epoch(self, epoch):
        self.model.train()
        if self.args['model'] == 'mme2e' or self.args['model'] == 'mme2e_sparse':
            self.model.mtcnn.eval()

        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        # with torch.autograd.set_detect_anomaly(True):
        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            # print(uttranceId)
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        epoch_loss /= len(dataloader.dataset)
        print(f'train loss = {epoch_loss}')
        # _, pred = torch.max(total_logits.data, 1)
        # _, truth = torch.max(total_Y.data, 1)
        # epoch_acc = 100 * (truth==pred).float().sum() / len(truth)
        epoch_preds = total_logits.argmax(-1)
        epoch_acc = accuracy_score(total_Y, epoch_preds)
        print(float(epoch_acc))

        self.train_writer.add_scalar('Loss', epoch_loss, epoch)
        self.train_writer.add_scalar('Accuracy', float(epoch_acc), epoch)

        return self.eval_func(total_logits, total_Y)
        # return eval_mosei_emo(total_logits, total_Y)


    # def train_one_epoch(self, epoch):
    #     phase = 'test'
    #     self.model.train()
    #     if self.args['model'] == 'mme2e' or self.args['model'] == 'mme2e_sparse':
    #         self.model.mtcnn.eval()

    #     dataloader = self.dataloaders['train']
    #     epoch_loss = 0.0
    #     data_size = 0
    #     total_logits = []
    #     epoch_loss_t = 0.0
    #     epoch_loss_v = 0.0
    #     epoch_loss_a = 0.0
    #     total_logits_t = []
    #     total_logits_v = []
    #     total_logits_a = []
    #     total_Y = []
    #     pbar = tqdm(dataloader, desc='Train')

    #     # with torch.autograd.set_detect_anomaly(True):
    #     for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
    #         # print(uttranceId)
    #         if 'lf_' not in self.args['model']:
    #             text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
    #         else:
    #             imgs = imgs.to(device=self.device)

    #         if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
    #             Y = Y.argmax(-1)

    #         # imgs = imgs.to(device=self.device)
    #         specgrams = specgrams.to(device=self.device)
    #         text = text.to(device=self.device)
    #         Y = Y.to(device=self.device)
    #         self.optimizer.zero_grad()
    #         with torch.set_grad_enabled(True):
    #             logits, logits_t, logits_v, logits_a = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
    #             loss = self.criterion(logits, Y)
    #             loss_t = self.criterion(logits_t, Y)
    #             loss_v = self.criterion(logits_v, Y)
    #             loss_a = self.criterion(logits_a, Y)
    #             loss.backward()
    #             epoch_loss += loss.item() * Y.size(0)

    #             epoch_loss_t += loss_t.item() * Y.size(0)
    #             epoch_loss_v += loss_v.item() * Y.size(0)
    #             epoch_loss_a += loss_a.item() * Y.size(0)

    #             data_size += Y.size(0)
    #             if self.args['clip'] > 0:
    #                 clip_grad_norm_(self.model.parameters(), self.args['clip'])
    #             self.optimizer.step()

    #             # if phase == 'test':
    #             #     logits_t, logits_v, logits_a = self.model.embeddings(imgs, imgLens, specgrams, specgramLens, text)
    #         total_logits.append(logits.cpu())
    #         total_Y.append(Y.cpu())
            
    #         if phase == 'test':
    #             total_logits_t.append(logits_t.cpu())
    #             total_logits_v.append(logits_v.cpu())
    #             total_logits_a.append(logits_a.cpu())
    #         pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
    #         if self.scheduler is not None:
    #             self.scheduler.step()
    #     total_logits = torch.cat(total_logits, dim=0)
    #     total_Y = torch.cat(total_Y, dim=0)
    #     epoch_loss /= len(dataloader.dataset)
    #     print(f'train loss = {epoch_loss}')

    #     epoch_loss_t /= len(dataloader.dataset)
    #     epoch_loss_v /= len(dataloader.dataset)
    #     epoch_loss_a /= len(dataloader.dataset)

    #     # _, pred = torch.max(total_logits.data, 1)
    #     # _, truth = torch.max(total_Y.data, 1)
    #     # epoch_acc = 100 * (truth==pred).float().sum() / len(truth)

    #     if phase == 'test':
    #         total_logits_t = torch.cat(total_logits_t, dim=0)
    #         total_logits_v = torch.cat(total_logits_v, dim=0)
    #         total_logits_a = torch.cat(total_logits_a, dim=0)
    #         epoch_preds_t = total_logits_t.argmax(-1)
    #         epoch_acc_t = accuracy_score(total_Y, epoch_preds_t)
    #         epoch_preds_v = total_logits_v.argmax(-1)
    #         epoch_acc_v = accuracy_score(total_Y, epoch_preds_v)
    #         epoch_preds_a = total_logits_a.argmax(-1)
    #         epoch_acc_a = accuracy_score(total_Y, epoch_preds_a)

    #     epoch_preds = total_logits.argmax(-1)
    #     epoch_acc = accuracy_score(total_Y, epoch_preds)
    #     print(float(epoch_acc))

    #     self.train_writer.add_scalar('Loss', epoch_loss, epoch)
    #     self.train_writer.add_scalar('Accuracy', float(epoch_acc), epoch)
    #     if phase == 'test':
    #         self.submodality_writer.add_scalars('Accuracy', {'Total Accuracy': float(epoch_acc),
    #                                 'Text Accuracy': float(epoch_acc_t),
    #                                 'Visual Accuracy': float(epoch_acc_v),
    #                                 'Audio Accuracy': float(epoch_acc_a),}, epoch)
    #         self.submodality_loss_writer.add_scalars('Loss', {'Total Loss': epoch_loss,
    #                                 'Text Loss': epoch_loss_t,
    #                                 'Visual Loss': epoch_loss_v,
    #                                 'Audio Loss': epoch_loss_a,}, epoch)

        # return self.eval_func(total_logits, total_Y)
        # # return eval_mosei_emo(total_logits, total_Y)

    def eval_one_epoch(self, epoch, phase='valid', thresholds=None):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                # logits, _, _, _ = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)

            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)

        # if phase == 'valid' and self.scheduler is not None:
        #     self.scheduler.step(epoch_loss)

        print(f'{phase} loss = {epoch_loss}')
        # _, pred = torch.max(total_logits.data, 1)
        # _, truth = torch.max(total_Y.data, 1)
        # epoch_acc = 100 * (truth==pred).float().sum() / len(truth)
        epoch_preds = total_logits.argmax(-1)
        epoch_acc = accuracy_score(total_Y, epoch_preds)
        print(float(epoch_acc))
        # target_names = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        # # target_names = ['ang-fru-neu-sad', 'neu-sad']
        # report = classification_report(total_Y, epoch_preds, target_names=target_names, output_dict=True)

        self.test_writer.add_scalar('Loss', epoch_loss, epoch)

        # return self.eval_func(total_logits, total_Y, thresholds)
        return self.eval_func(total_logits, total_Y)
        # return eval_mosei_emo(total_logits, total_Y, thresholds)

    # def eval_one_epoch(self, epoch, phase='valid', thresholds=None):
    #     self.model.eval()
    #     dataloader = self.dataloaders[phase]
    #     epoch_loss = 0.0
    #     data_size = 0
    #     total_logits = []
    #     total_logits_t = []
    #     total_logits_v = []
    #     total_logits_a = []
    #     total_Y = []
    #     pbar = tqdm(dataloader, desc=phase)

    #     for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
    #         if 'lf_' not in self.args['model']:
    #             text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
    #         else:
    #             imgs = imgs.to(device=self.device)

    #         if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
    #             Y = Y.argmax(-1)

    #         # imgs = imgs.to(device=self.device)
    #         specgrams = specgrams.to(device=self.device)
    #         text = text.to(device=self.device)
    #         Y = Y.to(device=self.device)

    #         with torch.set_grad_enabled(False):
    #             logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
    #             loss = self.criterion(logits, Y)
    #             epoch_loss += loss.item() * Y.size(0)
    #             data_size += Y.size(0)
    #             # print('logits: ', logits, '\n')

    #             if phase == 'test':
    #                 logits_t, logits_v, logits_a = self.model.embeddings(imgs, imgLens, specgrams, specgramLens, text)
    #                 # a = torch.transpose(self.model.weighted_fusion.weight, 0, 1)
    #                 # # print("model.weighted_fusion.weight: ", a)
    #                 # # print('shape1: ', a.shape)
    #                 # b = torch.stack([logits_t, logits_v, logits_a], dim=-1)
    #                 # # print('shape: ', b.shape)
    #                 # print("mul: ", torch.matmul(b, a))
    #                 # # torch.mm(model.weighted_fusion.weight, torch.stack(all_logits, dim=-1))
    #                 # print('logits_t: ', logits_t, 'logits_v: ', logits_v, 'logits_a: ', logits_t, '\n')

    #         total_logits.append(logits.cpu())
    #         total_Y.append(Y.cpu())
    #         if phase == 'test':
    #             total_logits_t.append(logits_t.cpu())
    #             total_logits_v.append(logits_v.cpu())
    #             total_logits_a.append(logits_a.cpu())

    #         pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

    #     total_logits = torch.cat(total_logits, dim=0)
    #     total_Y = torch.cat(total_Y, dim=0)

    #     if phase == 'test':
    #         total_logits_t = torch.cat(total_logits_t, dim=0)
    #         total_logits_v = torch.cat(total_logits_v, dim=0)
    #         total_logits_a = torch.cat(total_logits_a, dim=0)
    #         epoch_preds_t = total_logits_t.argmax(-1)
    #         epoch_acc_t = accuracy_score(total_Y, epoch_preds_t)
    #         epoch_preds_v = total_logits_v.argmax(-1)
    #         epoch_acc_v = accuracy_score(total_Y, epoch_preds_v)
    #         epoch_preds_a = total_logits_a.argmax(-1)
    #         epoch_acc_a = accuracy_score(total_Y, epoch_preds_a)

    #     epoch_loss /= len(dataloader.dataset)

    #     # if phase == 'valid' and self.scheduler is not None:
    #     #     self.scheduler.step(epoch_loss)

    #     print(f'{phase} loss = {epoch_loss}')
    #     # _, pred = torch.max(total_logits.data, 1)
    #     # _, truth = torch.max(total_Y.data, 1)
    #     # epoch_acc = 100 * (truth==pred).float().sum() / len(truth)
    #     epoch_preds = total_logits.argmax(-1)
    #     epoch_acc = accuracy_score(total_Y, epoch_preds)
    #     print(float(epoch_acc))
    #     # target_names = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    #     # # target_names = ['ang-fru-neu-sad', 'neu-sad']
    #     # report = classification_report(total_Y, epoch_preds, target_names=target_names, output_dict=True)

    #     self.test_writer.add_scalar('Loss', epoch_loss, epoch)
    #     self.test_writer.add_scalar('Accuracy', float(epoch_acc), epoch)
    #     if phase == 'test':
    #         self.submodality_writer.add_scalars('Evaluation', {'Total Accuracy': float(epoch_acc),
    #                                 'Text Accuracy': float(epoch_acc_t),
    #                                 'Visual Accuracy': float(epoch_acc_v),
    #                                 'Audio Accuracy': float(epoch_acc_a),}, epoch)

    #     # return self.eval_func(total_logits, total_Y, thresholds)
    #     return self.eval_func(total_logits, total_Y)
    #     # return eval_mosei_emo(total_logits, total_Y, thresholds)

    def test_one_epoch(self, phase='valid', thresholds=None):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        total_uttranceId = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)

            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            total_uttranceId.append(uttranceId)
            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        total_Id = []
        for i in total_uttranceId:
            total_Id.extend(i)

        epoch_loss /= len(dataloader.dataset)

        # if phase == 'valid' and self.scheduler is not None:
        #     self.scheduler.step(epoch_loss)

        # print(f'{phase} loss = {epoch_loss}')
        # return test_iemocap(total_logits, total_Y, total_Id, thresholds)
        return test_iemocap_ce(total_logits, total_Y)
        # return self.eval_func(total_logits, total_Y)
        # return eval_mosei_emo(total_logits, total_Y, thresholds)
    
    def test_one_epoch_hierachy(self, phase='test', thresholds=None):
        # model1 is 2_class(ang_fru_neu)(sad)
        args1 = self.args
        args1['num_emotions'] = 4
        model1 = MME2E_Sparse(args=args1, device=self.device)
        model1 = model1.to(device=self.device)
        model1.load_state_dict(torch.load('./savings/models/4_class_ang_fru_neu_sad_SCE_alpha=6.0_beta=0.1/mme2e_sparse_tav_0.5914_0.5914_0.6093_0.5985_imginvl500_st_0.7_seed0.pt'), strict=False)
        model1.eval()
        print('load 4_class_ang_fru_neu_sad successed..')

        args2 = self.args
        args2['num_emotions'] = 2
        model2 = MME2E_Sparse(args=args2, device=self.device)
        model2 = model2.to(device=self.device)
        model2.load_state_dict(torch.load('./savings/models/2_class_exc_hap_SCE_alpha=0.1_beta=1.0/mme2e_sparse_tav_0.6667_0.6531_0.6508_0.6518_imginvl500_st_0.7_seed0.pt'), strict=False)
        model2.eval()
        print('load 2_class_exc_hap successed..')

        # args3 = self.args
        # args3['num_emotions'] = 3
        # model3 = MME2E_Sparse(args=args2, device=self.device)
        # model3 = model3.to(device=self.device)
        # model3.load_state_dict(torch.load('./savings/models/3_class_ang_fru_neu/mme2e_sparse_tav_Acc_0.7347_F1_0.6440_AUC_0.7871_imginvl500_st_0.7_seed0.pt'), strict=False)
        # model3.eval()
        # print('load 3_class_ang_fru_neu successed..')

        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        total_uttranceId = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce' or self.args['loss'] == 'sce' or self.args['loss'] == 'focal' or self.args['loss'] == 'NCEandRCE':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                logits = logits.cpu()
                # print('logits: ', logits)
                pred_list = (np.argmax(logits, axis=1)).tolist()
                # print('pred_list: ', pred_list)
                # model labels=['ang-fru-neu-sad', 'exc-hap']
                if pred_list[0] == 0:
                    logits = model1(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                    logits = logits.cpu()
                    pred = (np.argmax(logits, axis=1)).tolist()
                    if pred[0] == 0:
                        logits = [1, 0, 0, 0, 0, 0] #anger
                    elif pred[0] == 1:
                        logits = [0, 0, 1, 0, 0, 0] #frustrated
                    elif pred[0] == 2:
                        logits = [0, 0, 0, 0, 1, 0] #neutral
                    else:
                        logits = [0, 0, 0, 0, 0, 1] #sadness
                else:
                    logits = model2(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                    logits = logits.cpu()
                    pred = (np.argmax(logits, axis=1)).tolist()
                    if pred[0] == 0:
                        logits = [0, 1, 0, 0, 0, 0] #excited
                    else:
                        logits = [0, 0, 0, 1, 0, 0] #happiness
                # loss = self.criterion(logits, Y)
                # epoch_loss += loss.item() * Y.size(0)
                # data_size += Y.size(0)

            total_logits.append(logits)
            [Y] = Y.cpu().numpy()
            total_Y.append(Y)
            total_uttranceId.append(uttranceId)
            # pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        # print('total_uttranceId: ', total_uttranceId)
        # print('total_logits: ', total_logits)
        # print('total_Y: ', total_Y)
        pred_list = (np.argmax(total_logits, axis=1)).tolist()
        truth_list = (np.argmax(total_Y, axis=1)).tolist()
        wrong_answer = pd.DataFrame(columns=['filename', 'predict', 'truth'])

        mat_confusion = confusion_matrix(truth_list, pred_list)
        print('mat_confusion: \n', mat_confusion)
            # 用 seaborn 畫
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        labels=['anger', 'excited', 'frustrated', 'happiness', 'neutral', 'sadness']
        # labels=['anger', 'frustrated']
        # labels=['ang-fru-neu', 'exc-hap', 'sad']
        sns.heatmap(
            mat_confusion, xticklabels=labels, yticklabels=labels,
            annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
        plt.title('Confusion Matrix', fontsize=15)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')
        plt.savefig('./savings/wrong_stat/confusion_matrix.png')


        # emotion_trans = {0: 'anger', 1: 'excited', 2: 'frustrated', 3: 'happiness', 4: 'neutral', 5: 'sadness'}
        # # emotion_trans = {0: 'anger', 1: 'frustrated'}
        # # emotion_trans = {0: 'ang-fru-neu', 1: 'exc-hap', 2: 'sad'}
        # for i in range(0, len(pred_list)):
        #     if pred_list[i] != truth_list[i]:
        #         print('len(wrong_answer): ', len(wrong_answer))
        #         wrong_answer.loc[len(wrong_answer)] = {'filename': uttranceId[i], 'predict': emotion_trans[pred_list[i]], 'truth': emotion_trans[truth_list[i]]}
        # wrong_answer.to_csv('./savings/wrong_stat/wrong.csv', index=False)
        # print('wrong_answer: \n', wrong_answer)