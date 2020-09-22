import numpy as np
import time
import argparse
import pickle
import os
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models import GKT, MultiHeadAttention, VAE
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
parser.add_argument('--data-file', type=str, default='skill_builder_data.csv', help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph', help='Where to load the pretrained dkt graph.')
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt', help='DKT graph data file name.')
parser.add_argument('--hid-dim', type=int, default=256, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=256, help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=128, help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=128, help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=128, help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--graph-type', type=str, default='Dense', help='The type of latent concept graph.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=5e-5, help='Output variance.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.7, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    save_dir = '{}/exp{}/'.format(args.save_dir, timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    gkt_file = os.path.join(save_dir, 'gkt.pt')
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
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(dataset_path, args.batch_size, args.graph_type,
                                                                           dkt_graph_path = dkt_graph_path,
                                                                           train_ratio=args.train_ratio,
                                                                           val_ratio=args.val_ratio,
                                                                           shuffle=args.shuffle, use_cuda=args.cuda)

# build model
att = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
vae = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim,
          edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)

graph_model = None  # if args.graph_type in : ['Dense', 'Transition', 'DKT', 'PAM']
if args.graph_type == 'MHA':
    graph_model = att
elif args.graph_type == 'VAE':
    graph_model = vae
gkt = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type, graph=graph, graph_model=graph_model,
          dropout=args.dropout, bias=args.bias)

kt_loss = KTLoss()
vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior)

if args.load_dir:
    gkt_file = os.path.join(args.load_dir, 'gkt.pt')
    gkt.load_state_dict(torch.load(gkt_file))
    args.save_dir = False

# build optimizer
optimizer = optim.Adam(gkt.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

if args.prior:
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
    if args.graph_type == 'MHA':
        att = att.cuda()
    if args.graph_type == 'VAE':
        vae = vae.cuda()
        vae_loss = vae_loss.cuda()
    gkt = gkt.cuda()
    kt_loss = KTLoss()


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    kt_train = []
    vae_train = []

    gkt.train()
    scheduler.step()
    for batch_idx, (features, questions, answers) in enumerate(train_loader):
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
        optimizer.zero_grad()
        pred_res, ec_list, rec_list, z_prob_list = gkt(features, questions)
        loss_kt = kt_loss(pred_res, answers)
        kt_train.append(loss_kt.item())
        loss = loss_kt
        if args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                vae_train.append(loss_vae.item())
            loss = loss + loss_vae
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()

    loss_val = []
    kt_val = []
    vae_val = []

    gkt.eval()
    for batch_idx, (features, questions, answers) in enumerate(valid_loader):
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
        pred_res, ec_list, rec_list, z_prob_list = gkt(features, questions)
        loss_kt = kt_loss(pred_res, answers)
        kt_val.append(loss_kt.item())
        loss = loss_kt
        if args.graph_type == 'VAE':
            loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
            vae_val.append(loss_vae.item())
            loss = loss + loss_vae
        loss_val.append(loss.item())

    if args.graph_type == 'VAE':
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'kt_train: {:.10f}'.format(np.mean(kt_train)),
              'vae_train: {:.10f}'.format(np.mean(vae_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'kt_val: {:.10f}'.format(np.mean(kt_val)),
              'vae_val: {:.10f}'.format(np.mean(vae_val)),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'time: {:.4f}s'.format(time.time() - t))
    if args.save_dir and np.mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')
        torch.save(gkt.state_dict(), gkt_file)
        if args.graph_type == 'VAE':
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'kt_train: {:.10f}'.format(np.mean(kt_train)),
                  'vae_train: {:.10f}'.format(np.mean(vae_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'kt_val: {:.10f}'.format(np.mean(kt_val)),
                  'vae_val: {:.10f}'.format(np.mean(vae_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        else:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(loss_val)


def test():
    loss_test = []
    kt_test = []
    vae_test = []

    gkt.eval()
    gkt.load_state_dict(torch.load(gkt_file))
    for batch_idx, (features, questions, answers) in enumerate(test_loader):
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
        pred_res, ec_list, rec_list, z_prob_list = gkt(features, questions)
        loss_kt = kt_loss(pred_res, answers)
        kt_test.append(loss_kt.item())
        loss = loss_kt
        if args.graph_type == 'VAE':
            loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
            vae_test.append(loss_vae.item())
            loss = loss + loss_vae
        loss_test.append(loss.item())

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    if args.graph_type == 'VAE':
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'kt_test: {:.10f}'.format(np.mean(kt_test)),
              'vae_test: {:.10f}'.format(np.mean(vae_test)))
    else:
        print('loss_test: {:.10f}'.format(np.mean(loss_test)))
    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        if args.graph_type == 'VAE':
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'kt_test: {:.10f}'.format(np.mean(kt_test)),
                  'vae_test: {:.10f}'.format(np.mean(vae_test)), file=log)
        else:
            print('loss_test: {:.10f}'.format(np.mean(loss_test)), file=log)
        log.flush()


# Train model
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