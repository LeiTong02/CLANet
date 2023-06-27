# CLANet: A Comprehensive Framework for Cross-Batch Cell Line Identification Using Brightfield Images.

<!-- ## Introduction -->
<div align="center">
  <img src="figs/framework.png"/>
</div><br/>
A time-series cell image sequence $\mathbb{X}_{s}$ is obtained from a single microscopy location within a flask. Each cell image $X_{n}$ undergoes the Cell Cluster-level Selection to generate patches $Q_{n}$. Patch embeddings are extracted from patches using self-supervised learning, forming the patch embedding sequence $\mathbb{F}_{s}$. During training, the Time-series Segment Sampling is applied to sample the patch embedding sequence into several snippets, which are then fed into a multiple instance learning (MIL) aggregator to compute the classification loss $\mathcal{L}_{cla}$. During the inference stage, the complete embedding sequence is directly passed into the MIL aggregator to obtain the predicted label.

## Preparation
This implementation is built upon pytorch==1.8.1+cuda111, the environment requirements can be configured:
```
pip install -r requirements.txt
```

* Environments  
  Linux, Python==3.6.8, CUDA == 11.2, pytorch == 1.9.0, mmdet3d == 0.17.1   

* Detection Data   
Follow the mmdet3d to process the nuScenes dataset (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).

* Segmentation Data  
Download Map expansion from nuScenes dataset (https://www.nuscenes.org/nuscenes#download). Extract the contents (folders basemap, expansion and prediction) to your nuScenes `maps` folder.  
Then build Segmentation dataset:
  ```
  cd tools
  python build-dataset.py
  ```
  

* After preparation, you will be able to see the following directory structure:  
  ```
  PETR
  ├── mmdetection3d
  ├── projects
  │   ├── configs
  │   ├── mmdet3d_plugin
  ├── tools
  ├── data
  │   ├── nuscenes
  │     ├── HDmaps-nocover
  │     ├── ...
  ├── ckpts
  ├── README.md
  ```

## Train & inference
<!-- ```bash
git clone https://github.com/megvii-research/PETR.git
``` -->
## Visualize


## Main Results  


## Acknowledgement
Many thanks to the authors of [dino](https://github.com/facebookresearch/dino), [SelectiveSearch](https://github.com/AlpacaTechJP/selectivesearch) and [AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py).


## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   

```

## Contact
If you have any questions, feel free to open an issue or contact us at lt228@leicester.ac.uk.
