# CLANet
A Comprehensive Framework for Cross-Batch Cell Line Identification Using Brightfield Images

<!-- ## Introduction -->
<div align="center">
  <img src="figs/framework.png"/>
</div><br/>
CLANet comprises three stages: (1) extracting significant cell patches from the time-series image sequence; (2) self-supervised learning to learn and extract feature embeddings from the extracted patches; (3) feature fusion using the MIL aggregator for predicting cell line identity.
<!--
## Preparation
This implementation is built upon [detr3d](https://github.com/WangYueFt/detr3d/blob/main/README.md), and can be constructed as the [install.md](./install.md).

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
  
  If you want to train the segmentation task immediately, we privided the processed data ( HDmaps-final.tar ) at [gdrive](https://drive.google.com/file/d/1uw-ciYbqEHRTR9JoGH8VXEiQGAQr7Kik/view?usp=sharing). The processed info files of segmentation can also be find at [gdrive](https://drive.google.com/drive/folders/1_C2yuh51ROF3UzId4L1itwGQVUeVUxU6?usp=sharing).


* Pretrained weights   
To verify the performance on the val set, we provide the pretrained V2-99 [weights](https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view?usp=sharing). The V2-99 is pretrained on DDAD15M ([weights](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth)) and further trained on nuScenes **train set** with FCOS3D.  For the results on test set in the paper, we use the DD3D pretrained [weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN). The ImageNet pretrained weights of other backbone can be found [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json).
Please put the pretrained weights into ./ckpts/. 

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
<!--
## Train & inference
<!-- ```bash
git clone https://github.com/megvii-research/PETR.git
``` -->
```bash
cd PETR
```
You can train the model following:
```bash
tools/dist_train.sh projects/configs/petr/petr_r50dcn_gridmask_p4.py 8 --work-dir work_dirs/petr_r50dcn_gridmask_p4/
```
You can evaluate the model following:
```bash
tools/dist_test.sh projects/configs/petr/petr_r50dcn_gridmask_p4.py work_dirs/petr_r50dcn_gridmask_p4/latest.pth 8 --eval bbox
```
## Visualize
-->

## Main Results
-->   


## Acknowledgement
Many thanks to the authors of [dino](https://github.com/facebookresearch/dino).


## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   

```

## Contact
If you have any questions, feel free to open an issue or contact us at lt228@leicester.ac.uk.
