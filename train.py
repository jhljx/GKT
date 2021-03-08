import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models import GKT, MultiHeadAttention, VAE, DKT
from metrics import KTLoss, VAELoss
from processing import load_dataset

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_false', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='data', help='Data dir for loading input data.')
parser.add_argument('--data-file', type=str, default='assistment_test15.csv', help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph', help='Where to load the pretrained dkt graph.')
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt', help='DKT graph data file name.')
parser.add_argument('--model', type=str, default='GKT', help='Model type to use, support GKT and DKT.')
parser.add_argument('--hid-dim', type=int, default=32, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=32, help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=32, help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=32, help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=32, help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--graph-type', type=str, default='Dense', help='The type of latent concept graph.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
parser.add_argument('--result-type', type=int, default=12, help='Number of results types when multiple results are used.')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT', help='Existed model file dir.')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

res_len = 2 if args.binary else args.result_type

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

# load dataset
dataset_path = os.path.join(args.data_dir, args.data_file)
dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
if not os.path.exists(dkt_graph_path):
    dkt_graph_path = None
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(dataset_path, args.batch_size, args.graph_type, dkt_graph_path=dkt_graph_path,
                                                                           train_ratio=args.train_ratio, val_ratio=args.val_ratio, shuffle=args.shuffle,
                                                                           model_type=args.model, use_cuda=args.cuda)

# build models
graph_model = None
if args.model == 'GKT':
    if args.graph_type == 'MHA':
        graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
    elif args.graph_type == 'VAE':
        graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
                          edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)
        vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior, var=args.var)
        if args.cuda:
            vae_loss = vae_loss.cuda()
    if args.cuda:
        graph_model = graph_model.cuda()
    model = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type, graph=graph, graph_model=graph_model,
                dropout=args.dropout, bias=args.bias, res_len=res_len)
elif args.model == 'DKT':
    model = DKT(res_len * concept_num, args.emb_dim, concept_num, dropout=args.dropout, bias=args.bias)
else:
    raise NotImplementedError(args.model + ' model is not implemented!')
kt_loss = KTLoss()

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

# load model/optimizer/scheduler params
if args.load_dir:
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

if args.model == 'GKT' and args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    kt_train = []
    vae_train = []
    auc_train = []
    acc_train = []
    if graph_model is not None:
        graph_model.train()
    model.train()
    for batch_idx, (features, questions, answers) in enumerate(train_loader):
        t1 = time.time()
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
        ec_list, rec_list, z_prob_list = None, None, None
        if args.model == 'GKT':
            pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
        elif args.model == 'DKT':
            pred_res = model(features, questions)
        else:
            raise NotImplementedError(args.model + ' model is not implemented!')
        loss_kt, auc, acc = kt_loss(pred_res, answers)
        kt_train.append(float(loss_kt.cpu().detach().numpy()))
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)

        if args.model == 'GKT' and args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                vae_train.append(float(loss_vae.cpu().detach().numpy()))
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'loss vae: ', loss_vae.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
            loss = loss_kt + loss_vae
        else:
            loss = loss_kt
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
        loss_train.append(float(loss.cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        del loss
        print('cost time: ', str(time.time() - t1))

    loss_val = []
    kt_val = []
    vae_val = []
    auc_val = []
    acc_val = []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')
            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            kt_val.append(loss_kt)
            if auc != -1 and acc != -1:
                auc_val.append(auc)
                acc_val.append(acc)

            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_val.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_val.append(loss)
            del loss
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'kt_train: {:.10f}'.format(np.mean(kt_train)),
              'vae_train: {:.10f}'.format(np.mean(vae_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'kt_val: {:.10f}'.format(np.mean(kt_val)),
              'vae_val: {:.10f}'.format(np.mean(vae_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    if args.save_dir and np.mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)
        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'kt_train: {:.10f}'.format(np.mean(kt_train)),
                  'vae_train: {:.10f}'.format(np.mean(vae_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'kt_val: {:.10f}'.format(np.mean(kt_val)),
                  'vae_val: {:.10f}'.format(np.mean(vae_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            del kt_train
            del vae_train
            del kt_val
            del vae_val
        else:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    res = np.mean(loss_val)
    del loss_train
    del auc_train
    del acc_train
    del loss_val
    del auc_val
    del acc_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res


def test():
    loss_test = []
    kt_test = []
    vae_test = []
    auc_test = []
    acc_test = []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(test_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')
            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
            kt_test.append(loss_kt)
            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_test.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_test.append(loss)
            del loss
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'kt_test: {:.10f}'.format(np.mean(kt_test)),
              'vae_test: {:.10f}'.format(np.mean(vae_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
    else:
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'kt_test: {:.10f}'.format(np.mean(kt_test)),
                  'vae_test: {:.10f}'.format(np.mean(vae_test)),
                  'auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
            del kt_test
            del vae_test
        else:
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
        log.flush()
    del loss_test
    del auc_test
    del acc_test
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()

if args.test is False:
    # Train model
    print('start training!')
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_dir:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

test()
if log is not None:
    print(save_dir)
    log.close()