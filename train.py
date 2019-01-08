################### LIBRARIES ###################
import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json

import matplotlib
matplotlib.use('agg')

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import netlib as netlib


################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()
parser.add_argument('--lr',           default=0.0005, type=float, help='Learning Rate')
parser.add_argument('--perc_data',    default=1,      type=float, help='Percentage of Data to Use')
parser.add_argument('--tv_split',     default=0.6,   type=float, help='Train-Validation Split')
parser.add_argument('--n_epochs',     default=100,    type=int, help='Number of training epochs.')
parser.add_argument('--tau',          default=60,     type=int, help='Stepsize before reducing learning rate.')
parser.add_argument('--num_classes',  default=200,    type=int, help='Obsolete flag to set number of output units in Neural Network when using another loss.')
parser.add_argument('--gamma',        default=0.1,    type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--kernels',      default=8,      type=int, help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',           default=16,     type=int, help='Mini-Batchsize to use.')
parser.add_argument('--seed',         default=1,      type=int, help='Random seed for reproducibility.')
parser.add_argument('--dataset',      default='cub200',   type=str, help='Dataset to use.')
parser.add_argument('--savename',     default='',   type=str, help='Appendix to save folder name if any special information is to be included.')
parser.add_argument('--source_path',  default='<Path_where_all_dataset_are_located>', type=str, help='Path to training data.')
parser.add_argument('--save_path',    default='<Path_where_to_save_network_weights>', type=str, help='Where to save everything.')
opt = parser.parse_args()

opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset+'_TripletTraining'

#########################################################
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
rng = np.random.RandomState(opt.seed)



#################### DATALOADER SETUPS ##################
if opt.dataset=='cars196': train_dataset, val_dataset = aux.give_CARS196_training_sets(opt.source_path, opt.perc_data, opt.tv_split, opt.seed)
if opt.dataset=='celeba':  train_dataset, val_dataset = aux.give_CelebA_training_sets(opt.source_path, opt.perc_data, opt.tv_split, opt.seed)
if opt.dataset=='cub200':  train_dataset, val_dataset = aux.give_CUB200_training_sets(opt.source_path, opt.perc_data, opt.tv_split, opt.seed)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=opt.bs, num_workers=opt.kernels, pin_memory=True)


#################### CREATE SAVING FOLDER ###############
date = datetime.datetime.now()
time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
checkfolder = opt.save_path+'/_{}_TripleNet_'.format(opt.dataset)+time_string
counter     = 1
while os.path.exists(checkfolder):
    checkfolder = opt.save_path+'_'+str(counter)
    counter += 1
os.makedirs(checkfolder)
opt.save_path = checkfolder

#################### SAVE OPTIONS TO TXT ################
with open(opt.save_path+'/Parameter_Info.txt','w') as f:
    option_dict = vars(opt)
    json.dump(option_dict, f)
pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))

#################### CREATE LOGGING FILES ###############
InfoPlotter   = aux.InfoPlotter(opt.save_path+'/InfoPlot.svg')
CSV_log_train = aux.CSV_Writer(opt.save_path+'/log_epoch_train.csv', ['Epoch', 'Loss', 'Time'])
CSV_log_val   = aux.CSV_Writer(opt.save_path+'/log_epoch_val.csv', ['Epoch', 'Loss', 'Time'])
Progress_Saver= {'Train Loss':[], 'Val Loss':[]}


#################### SETUP TRIPLENET ###################
opt.device = torch.device('cuda')
model = netlib.FC_AlexNet(opt.num_classes)
print('TripletNet Setup complete with #weights: {}'.format(aux.gimme_params(model)))
_ = model.to(opt.device)


#################### TRAINING SETUP ####################
triplet_loss = netlib.TripletLoss()
optimizer    = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.tau, gamma=opt.gamma)
global best_val_loss
best_val_loss = np.inf

#################### TRAINER & EVALUATION FUNCTIONS ############################
def train_one_epoch(dataloader, model, optimizer, triplet_loss, opt, progress_saver, epoch):
    dataloader = train_dataloader
    _ = model.train()

    loss_collect = []

    start = time.time()
    for i,(anchor_img, pos_img, neg_img) in enumerate(tqdm(dataloader, desc='EPOCH Training ')):
        # anchor_img, pos_img, neg_img = next(iter(dataloader))
        anchor_img, pos_img, neg_img = anchor_img.to(opt.device), pos_img.to(opt.device), neg_img.to(opt.device)

        features = model(anchor_img, pos_img, neg_img)
        loss     = triplet_loss(*features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_collect.append(loss.item())

    CSV_log_train.log([epoch, np.mean(loss_collect), np.round(time.time()-start,4)])
    progress_saver['Train Loss'].append(np.mean(loss_collect))


def evaluate(dataloader, model, optimizer, triplet_loss, opt, progress_saver, epoch):
    global best_val_loss
    _ = model.eval()

    loss_collect = []

    start = time.time()
    for i,(anchor_img, pos_img, neg_img) in enumerate(tqdm(dataloader, desc='EPOCH Validation')):
        anchor_img, pos_img, neg_img = anchor_img.to(opt.device), pos_img.to(opt.device), neg_img.to(opt.device)

        features = model(anchor_img, pos_img, neg_img)
        loss     = triplet_loss(*features)

        loss_collect.append(loss.item())

    if np.mean(loss_collect)<best_val_loss:
        set_checkpoint(model, epoch,  optimizer, opt.save_path, progress_saver)

    CSV_log_val.log([epoch, np.mean(loss_collect), np.round(time.time()-start)])
    progress_saver['Val Loss'].append(np.mean(loss_collect))


def set_checkpoint(model, epoch, optimizer, savepath, progress_saver):
    torch.save({'epoch': epoch+1, 'state_dict':model.state_dict(),
                'optim_state_dict':optimizer.state_dict(), 'opt':opt,
                'progress':progress_saver}, savepath+'/checkpoint.pth.tar')




################### SCRIPT MAIN ##########################
for epoch in tqdm(range(opt.n_epochs), desc='Progress'):
    ### Train one epoch
    train_one_epoch(train_dataloader, model,optimizer, triplet_loss, opt, Progress_Saver, epoch)
    ### Validate
    evaluate(val_dataloader, model, optimizer, triplet_loss, opt, Progress_Saver, epoch)
    ### Create Summary Plot
    InfoPlotter.make_plot(range(epoch+1), Progress_Saver['Train Loss'], Progress_Saver['Val Loss'])
