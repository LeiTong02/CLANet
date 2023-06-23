# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from natsort import natsorted
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import numpy as np
from cell_dataset import Cell_Images,Cell_masked_patches
import utils
import vision_transformer as vits
import timm

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    elif "resnet" in args.arch:
        model = timm.create_model('resnet50', pretrained=False, num_classes=32)
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)


    model.cuda()
    model.eval()
    # load weights to evaluate
    if "resnet" in args.arch:
        print('==> Resuming from checkpoint..')
        # change here
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['net'])
        feature_net = torch.nn.Sequential(*(list(model.module.children())[:-1]))
        model = feature_net


    else:
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")
    linear_classifier = LinearClassifier(1053, num_labels=args.num_labels)
    #linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    if args.output_level == 'features':
        pass
    else:
        checkpoint = torch.load(os.path.join(args.output_dir,"checkpoint.pth.tar"), map_location="cpu")
        linear_classifier.load_state_dict(checkpoint['state_dict'])
    linear_classifier.eval()

    if args.sample_level == 'image':
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(224),
            pth_transforms.ToTensor(),
        ])


    elif args.sample_level == 'patch':
        train_transform = pth_transforms.Compose([

            pth_transforms.ToTensor(),
        ])

    '''engineered cells'''
    # data_path = '/projects/img/cellbank/Cell_Batch_Classification/Engineered_Cells/'
    # patch_npy_root = '/projects/img/cellbank/Cell_Batch_Classification/enhanced_engineered_cells_mask'
    # label_id_path_file = '/projects/img/cellbank/Cell_Batch_Classification/model/classification/train_engineered_class_idx.txt'

    data_path = "/projects/img/cellbank/Cell_lines/"
    patch_npy_root = "../image_masks/"
    label_id_path_file = '../train_32class_batch_idx.txt'


    with open(label_id_path_file, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    cell_folder_names = natsorted(lines)[:]

    print('we have {} cell lines'.format(len(cell_folder_names)))



    for folder_name in cell_folder_names[:]:
        batch_number = len(os.listdir(os.path.join(data_path, folder_name)))
        for i in range(batch_number, batch_number + 1, 1):
            print('Load {} batch_{}'.format(folder_name,i))
            if args.sample_level == 'image':
                dataset = Cell_Images(data_path, label_id_path_file, train=False, sup_train=False,
                                              transform=train_transform,
                                              selected_cell_data=folder_name, batch_id=i)
            elif args.sample_level == 'patch':
                dataset = Cell_masked_patches(data_path, label_id_path_file, train=False, sup_train=False, transform=train_transform,
                                              selected_cell_data=folder_name,batch_id=i,patch_npy_root=patch_npy_root)

            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=0
            )

            unique_image_tokens = np.array(dataset.unique_image_tokens).astype(int)

            patch_path = os.path.join(patch_npy_root, folder_name,'batch_{}'.format(i), "patches.npy")
            patch_dict = np.load(patch_path, allow_pickle=True).item()

            patch_representations = patch_dict.copy()
            #img_names = natsorted(os.listdir( os.path.join(data_path,folder_name, 'batch_{}'.format(i))))
            img_names = natsorted(patch_dict.keys())
            # evaluate

            predict_results = validate_network(val_loader, model, linear_classifier
                                               , args.n_last_blocks, args.avgpool_patchtokens,args.output_level)

            if args.sample_level == 'image':
                patch_representations={}
                for ids, i_name in enumerate(img_names):
                    #token_index = np.where(unique_image_tokens == ids)[0]
                    patch_representations[i_name] = predict_results[ids]
                if args.output_level == 'features':
                    patch_representation_path = os.path.join(patch_npy_root, folder_name, 'batch_{}'.format(i),
                                                             "image_features.npy")
                else:
                    patch_representation_path = os.path.join(patch_npy_root, folder_name, 'batch_{}'.format(i),
                                                             "image_predict_dino.npy")
                np.save(patch_representation_path, patch_representations)


            elif args.sample_level == 'patch':
                for ids, i_name in enumerate(img_names):
                    token_index = np.where(unique_image_tokens == ids)[0]

                    if len(patch_representations[i_name])>1:

                        for r_i in range(len(patch_representations[i_name])):
                            patch_representations[i_name][r_i]['prediction'] = predict_results[token_index[r_i]]

                if args.output_level == 'features':
                    patch_representation_path = os.path.join(patch_npy_root, folder_name, 'batch_{}'.format(i),
                                                             "patch_features.npy")
                else:
                    patch_representation_path = os.path.join(patch_npy_root, folder_name,'batch_{}'.format(i), "patch_predict_dino.npy")
                np.save(patch_representation_path,patch_representations)


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool,output_level):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    predict_results = []
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        if output_level == 'features':
            predict_results.append(output)
        else:
            output = linear_classifier(output)
            predict_results.append(output)

    predict_results = np.array(torch.cat(predict_results, dim=0).cpu().numpy())

    return predict_results


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./exp_patch/checkpoint_main.pth', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--sample_level", default="patch", type=str,
                        help='patch or image')
    parser.add_argument("--output_level", default="features", type=str,
                        help='features or classifier output')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./exp_patch_cla", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=32, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)
