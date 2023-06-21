
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.functional import softmax
import numpy as np
import torchvision.transforms as transforms
import time
import os
import argparse
import sys
sys.path.append("")
from datasets import Cell_Sampling_new
from models import GatedAttention

from Utils import progress_bar, Logger, mkdir_p,DataParallel_withLoss
from sklearn.metrics import top_k_accuracy_score,f1_score,roc_auc_score


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='attention', type=str, help='choosing network')
parser.add_argument('--data', default='AZCell', type=str, help='choosing network')
parser.add_argument('--bs', default=32, type=int, help='batch size')
'''Trianing_epochs = args.es* args.repeats'''
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--repeats', default=20, type=int, help='repeat training')
parser.add_argument('--split', default='batch_separated', type=str, help='dataset split type')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')

args = parser.parse_args()

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    weights = [item[2] for item in batch]

    return [data,target,weights]


def main(random_script_id):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    # Data
    print('==> Preparing data..')


    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    '''self-supverised learning features'''

    data_path = './image_masks/'
    label_id_path_file = './train_32class_batch_idx.txt'


    random_train_val = './experiment_setup/{}/random{}.txt'.format(args.split,random_script_id)

    if args.split == 'batch_separated':
        print('batch_separated Dataset')
        train_set = Cell_Sampling_new.Cell_Features_sampling(data_path,label_id_path_file,train=True,train_epoch=args.es*args.repeats,
                                  transform=transform_test,random_train_val=random_train_val)

        val_set  = Cell_Sampling_new.Cell_Features_sampling(data_path,label_id_path_file,train=False,train_epoch=args.es*args.repeats,
                                 transform=transform_test,random_train_val=random_train_val)
    else:
        print('batch_stratified Dataset')
        train_set = Cell_Sampling_new.Cell_Features_sampling_batch_stratified(data_path, label_id_path_file, train=True,train_epoch=args.es*args.repeats,
                                                             transform=transform_test, random_train_val=random_script_id)

        val_set = Cell_Sampling_new.Cell_Features_sampling_batch_stratified(data_path, label_id_path_file, train=False,train_epoch=args.es*args.repeats,
                                                           transform=transform_test, random_train_val=random_script_id)



    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0,collate_fn=my_collate)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0,collate_fn=my_collate)


    train_class_num = train_set.class_number


    if args.split == 'batch_separated':
        args.checkpoint = './experimental_models/batch_separated/random{}/CLANet/'.format(random_script_id)
    else:
        args.checkpoint = './experimental_models/batch_stratified/random{}/CLANet/'.format(random_script_id)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Model
    net = GatedAttention(train_class_num)

    #net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')


    if device == 'cuda':

        net = DataParallel_withLoss(net,criterion,device_ids=[0],output_device=0)

        cudnn.benchmark = True


    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):

            print('==> Resuming from checkpoint..')
            # change here
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])

            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))


        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss','Train Acc.','Val Loss','Val Acc Top1-5/F1.','Val Batch Acc'])





    val_loss_pre = 9999999
    if not args.evaluate:
        start_time = time.time()
        for epoch in range(start_epoch, start_epoch + args.es*args.repeats):

            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))

            train_loss, train_acc = train(net,trainloader,optimizer,criterion,device,)
            # Validation
            val_seq_accs, val_batch_accs = 0, 0
            val_loss, val_acc = 0, 0

            if (epoch+1)%(5)==0:
                val_loss, val_seq_accs, val_batch_accs = Validation(net, valloader, val_set.batch_token,train_class_num)

                print('\n Validation Loss: {}, Validation seq acc: {}'.format(val_loss,val_seq_accs,val_batch_accs))
                if val_loss < val_loss_pre:
                    print('-------val loss decrease, save best model now-----------')
                    val_loss_pre = val_loss
                    save_model(net, None, epoch, os.path.join(args.checkpoint,'best_model.pth'))

            logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc,val_loss,val_seq_accs, val_batch_accs])

        save_model(net, None, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
        print('------------Training Finished-----------------!')


    logger.close()


# Training
def train(net,trainloader,optimizer,criterion,device,):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs,targets,weights) in enumerate(trainloader):
        weights_sum = sum(weights).cuda()
        for i in range(len(inputs)):
            input_cuda,target,weight = inputs[i].cuda(),targets[i].cuda(),weights[i].cuda()


            loss, outputs,_ = net(target,input_cuda)

            loss = loss.sum()*weight/weights_sum


            loss.backward()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if ((i+1)%len(inputs)==0):
                optimizer.step()
                optimizer.zero_grad()


        progress_bar(batch_idx, len(trainloader), 'Loss: %.7f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total
#Validation
def Validation(net,valloader,batch_token,class_num):
    net.eval()

    val_loss = 0
    scores, labels = [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(valloader):
            inputs, targets = inputs[0].cuda(), targets[0].cuda()

            loss,outputs,_ = net(targets,inputs)
            loss = loss.sum()

            val_loss += loss.item()

            scores.append(softmax(outputs))
            labels.append(targets)

    scores = np.array(torch.cat(scores, dim=0).cpu().numpy())
    labels = np.array(torch.cat(labels, dim=0).cpu().numpy())

    seq_top1_acc,seq_top3_acc,seq_top5_acc = top_k_accuracy_score(labels,scores,k=1,labels=range(class_num))\
                                            ,top_k_accuracy_score(labels,scores,k=3,labels=range(class_num))\
                                            ,top_k_accuracy_score(labels,scores,k=5,labels=range(class_num))

    seq_f1 = f1_score(labels,np.argmax(scores,axis=1),average='macro')
    seq_auc = roc_auc_score(labels,scores,multi_class='ovr')

    b_scores, b_labels = [],[]
    for batch_id in np.unique(batch_token):
        b_ids = np.where(batch_token == batch_id)[0]

        b_score, b_label = np.mean(scores[b_ids],axis=0), np.argmax(np.bincount(labels[b_ids]))

        b_scores.append(b_score)
        b_labels.append(b_label)

    b_top1_acc, b_top3_acc, b_top5_acc = top_k_accuracy_score(b_labels, b_scores, k=1, labels=range(class_num)) \
                                        , top_k_accuracy_score(b_labels, b_scores, k=3, labels=range(class_num)) \
                                        , top_k_accuracy_score(b_labels, b_scores, k=5, labels=range(class_num))

    b_f1 = f1_score(b_labels, np.argmax(b_scores, axis=1), average='macro')
    b_auc = roc_auc_score(b_labels, b_scores, multi_class='ovr')

    return val_loss/(batch_idx+1), (seq_top1_acc,seq_top3_acc,seq_top5_acc,seq_f1,seq_auc), (b_top1_acc, b_top3_acc, b_top5_acc,b_f1,b_auc )


def save_model(net, acc, epoch, path):


    print('Saving..')
    state = {
        'net': net.module.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':

    for i in [1,2,3]:
        main(i)



