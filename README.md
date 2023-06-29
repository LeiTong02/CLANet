# CLANet: A Comprehensive Framework for Cross-Batch Cell Line Identification Using Brightfield Images.
CLANet is a pioneering framework for cross-batch cell line identification using brightfield images, specifically designed to tackle three distinct batch effects. The framework comprises three stages: (1) extracting significant cell patches from the time-series image sequence; (2) self-supervised learning to learn and extract feature embeddings from the extracted patches; (3) feature fusion using the MIL aggregator for predicting cell line identity.

<!-- ## Introduction -->
<div align="center">
  <img src="figs/framework.png"/>
</div><br/>

A time-series cell image sequence $\mathbb{X}\_{s}$ is obtained from a single microscopy location within a flask. Each cell image $X\_{n}$ undergoes the Cell Cluster-level Selection to generate patches $Q\_{n}$. Patch embeddings are extracted from patches using self-supervised learning, forming the patch embedding sequence $\mathbb{F}\_{s}$. During training, the Time-series Segment Sampling is applied to sample the patch embedding sequence into several snippets, which are then fed into a multiple instance learning (MIL) aggregator to compute the classification loss $\mathcal{L}\_{cla}$. During the inference stage, the complete embedding sequence is directly passed into the MIL aggregator to obtain the predicted label.

## Preparation
This implementation is built upon pytorch==1.8.1+cuda111, the environment can be configured:
```
pip install -r requirements.txt
```
## Dataset
The Dataset is currently undergoing internal ethical checking at AstraZeneca company and will be published soon.
* After the data preparation, you will find the following directory structure:  
  ```
  Cell_lines
  ├── A427 (cell line name)
  ├── A549
  │   ├── batch_1
  │     ├── cell_image1.png
  │     ├── cell_image2.png
  │     ├── ...
  │   ├── batch_2
  │   ├── ...
  ├── ...
  ```
Each cell image in the dataset follows a naming convention such as "VID1709\_G7\_1\_02d12h00m.jpg". Here, "VID1709" represents the index of the biological experiments (not relevant to this work), "G7" indicates the flask position that helps us split samples into sequences (or bags) in MIL (Multiple Instance Learning), "1" is the flask ID (not used), and "02d12h00m" denotes the incubation timepoint. This timepoint means that the image was collected on the 2nd day, 12 hours, and 00 minutes.

## 1. Patch Selection
The patch selection stage involves two steps: generating binary masks for cell images and performing cell cluster-level selection.
* Generate binary Masks for cell images (please modify the input path to the data path):
  ```
  python generate_binary_masks.py
  ```
* Cell cluster-level selection:
  ```
  python cell_cluster_selection.py
  ```
Once these steps are completed, you will find the following directory structure for the masks (similar to your data folder structure):
  ```
  image_masks
  ├── A427
  ├── A549
  │   ├── batch_1
  │     ├── cell_mask1.png
  │     ├── cell_mask2.png
  │     ├── patches.npy (patch locations)
  │     ├── ...
  │   ├── batch_2
  │   ├── ...
  ├── ...
  ```

## 2. Self-Supervised Learning for Feature Embedding
To train the self-supervised learning (SSL) model with the ViT-small network on a single node using 2 NVIDIA Tesla V100 GPUs 32GB for 200 epochs, use the following commands:
```
cd dino
python -m torch.distributed.launch --nproc_per_node=2 main_dino_patch.py --arch vit_small --batch_size_per_gpu 128 --epochs 200
```
To extract feature embeddings from the trained model, execute the following command:
```
python -m torch.distributed.launch --nproc_per_node=2 get_patch_prediction.py --sample_level patch --output_level features
```
The extracted patch features will be saved in the following directory structure:
```
./image_masks/cell_line_name/batch_ID/patch_features.npy
```
## 3. Multiple Instance Learning for Feature Fusion
After completing the previous steps, the TSS\_MIL can be trained using the following command:
```
python train_TSS_MIL.py --split batch_separated
```
The --split option specifies the evaluation tasks for the model. The evaluation results can be found in the following directory:
```
./experimental_models/selected_evaluation_task/random_ID/CLANet/log.txt
```
The results are presented as (top 1 acc, top 3 acc, top 5 acc, F1 score, AUC) for both sequence and batch levels in the validation columns.

Alternatively, you can run the jupyter script [model_evaluate.ipynb](https://github.com/LeiTong02/CLANet/blob/main/model_evaluate.ipynb) to display the performance on the test set.

## Main Results:
We introduce two strategies for splitting the data into training and test sets.
* Batch-separated: We randomly select 1 batch of data from each of the 32 cell classes as the training set, while the remaining 61 batches are assigned to the test set. In this way, the experimental batches in the test set differ from those in the training set, enabling the evaluation of cross-batch generalization (i.e. out-of-domain generalization) performance.
  |     Model     | Sequence\_Acc | Batch\_Acc | Sequence\_F1 | Batch\_F1 |
  |:-------------:|--------------|-----------|-------------|----------|
  | CLANet (ours) |     89.1%    |   90.7%   |    83.7%    | 87.0%    |
* Batch-stratified: In this strategy, both the training and test sets contain data from all experimental batches. The data are split within each batch, forming the sets. The sizes of the training and test sets are made roughly the same as in the batch-separated split. This split strategy is used to explore the influence of batch effects on classification performance in comparison to other methods.
  |     Model     | Sequence\_Acc | Batch\_Acc | Sequence\_F1 | Batch\_F1 |
  |:-------------:|--------------|-----------|-------------|----------|
  | CLANet (ours) |     99.9%    |   100.0%   |    99.9%    | 100.0%    |
Please refer to the [experiment_setup](https://github.com/LeiTong02/CLANet/tree/main/experiment_setup) directory in the GitHub repository for more details on the random split implementation.

## Acknowledgement
Many thanks to the authors of [dino](https://github.com/facebookresearch/dino), [SelectiveSearch](https://github.com/AlpacaTechJP/selectivesearch) and [AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py).


## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   

```

## Contact
If you have any questions, feel free to open an issue or contact us at lt228@leicester.ac.uk.
