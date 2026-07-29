import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.datasets import get_dataset_iemocap, collate_fn, HCFDataLoader, get_dataset_mosei, collate_fn_hcf_mosei
# from src.models.e2e import MME2E
from src.models.sparse_e2e import MME2E_Sparse
from src.models.e2e import MME2E
from src.models.baselines.lf_rnn import LF_RNN
from src.models.baselines.lf_transformer import LF_Transformer
from src.trainers.emotiontrainer import IemocapTrainer
from src.loss import SCELoss, FocalLoss
from src.loss_1 import NCEandRCE
from src.ib_losses import IB_FocalLoss
from src.sampler import ClassAwareSampler
import warnings
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler

if __name__ == "__main__":
    start = time.time()

    args = get_args()
    warnings.filterwarnings('ignore')
    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Current allocated memory:", torch.cuda.memory_allocated())


    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='valid',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test',
                                           img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        # train_sampler = ClassAwareSampler(train_dataset)
        # print('Using Class Aware Sampler...')

        if args['hand_crafted']:
            train_loader = HCFDataLoader(dataset=train_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=True, num_workers=2)
            valid_loader = HCFDataLoader(dataset=valid_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=False, num_workers=2)
            test_loader = HCFDataLoader(dataset=test_dataset, feature_type=args['audio_feature_type'],
                                        batch_size=args['batch_size'], shuffle=False, num_workers=2)
        else:
            # train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], sampler=train_sampler,
            #                           num_workers=2, collate_fn=collate_fn)
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=2, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False,
                                      num_workers=2, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                     num_workers=2, collate_fn=collate_fn)
    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']
    if args['model'] == 'mme2e':
        model = MME2E(args=args, device=device)
        model = model.to(device=device)
        # # model.load_state_dict(torch.load('./savings/models/all_single_label_six_category/with_valid/pretrained_no_mosei_no_MELD_12/mme2e_tav_0.6252_0.6164_0.6595_0.6340_imginvl500_seed0.pt'), strict=False)
        # model.load_state_dict(torch.load('/home/matt/Final_result/without_LIRIS/models/pretrain_no_mosei_no_IEMOCAP_LIRIS(four_class)/mme2e_tav_0.6866_0.6863_0.6971_0.6787_imginvl500_seed0.pt'), strict=False)
        # print('load pretrained model pretrain_no_mosei_no_IEMOCAP_LIRIS(four_class) (mme2e_tav_0.6866_0.6863_0.6971_0.6787_imginvl500_seed0) successed..')
        # # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts

        # for name, param in model.named_parameters():
        #     print('name: ', name)
        #     print('requires_grad: ', param.requires_grad)
        # for name, param in model.named_parameters():
        #     if name not in ['v_out.weight', 'v_out.bias', 't_out.weight', 't_out.bias', 'a_out.weight', 'a_out.bias', 'weighted_fusion.weight']:
        #         param.requires_grad = False
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
            # parameters = filter(lambda p: p.requires_grad, model.parameters())
            # optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'mme2e_sparse':
        model = MME2E_Sparse(args=args, device=device)
        model = model.to(device=device)


        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_rnn':
        model = LF_RNN(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_transformer':
        model = LF_Transformer(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    else:
        raise ValueError('Incorrect model name!')

    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size'])
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12], gamma=0.1)
    else:
        scheduler = None
    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif args['loss'] == 'ce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        pos_weight = pos_weight.to(torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=pos_weight)
        # criterion = torch.nn.CrossEntropyLoss()
    elif args['loss'] == 'sce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        pos_weight = pos_weight.to(torch.float32)
        criterion = SCELoss(weight=pos_weight, alpha=args['alpha'], beta=args['beta'], num_classes=args['num_emotions'])
    elif args['loss'] == 'NCEandRCE':
        criterion = NCEandRCE(alpha=6, beta=0.1, num_classes=6)
        print('NCEandRCE: alpha = 6, beta = 0.1')
    elif args['loss'] == 'focal':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        pos_weight = pos_weight.to(torch.float32)
        # criterion = FocalLoss(weight=pos_weight, gamma=1)
        criterion = IB_FocalLoss(weight=pos_weight, alpha=1000, gamma=1)
        # # ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # ce_loss = torch.nn.functional.cross_entropy(reduction='none')
        # pt = torch.exp(-ce_loss)
        # criterion = (0.25* (1-pt)**2 * ce_loss).mean()
        # criterion = FocalLoss(gamma=2, with_logits=True, reduction='mean')
    elif args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = torch.nn.BCEWithLogitsLoss()

    if args['dataset'] == 'iemocap' or 'mosei':
        trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    end = time.time()

    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
