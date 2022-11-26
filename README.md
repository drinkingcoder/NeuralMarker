# NeuralMarker
### [Project Page](https://drinkingcoder.github.io/publication/neuralmarker/)

> NeuralMarker: A Framework for Learning General Marker Correspondence   
> [Zhaoyang Huang](https://drinkingcoder.github.io)<sup>\*</sup>, Xiaokun Pan<sup>\*</sup>, Weihong Pan, Weikang Bian, [Yan Xu](https://decayale.github.io/), Ka Chun Cheung, [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)  
> SIGGRAPH Asia (ToG) 2022  

## TODO List
- [x] Code release (2022-10-15)
- [ ] Models release (2022-10-20)
- [ ] Demo code release (2022-10-25)
- [ ] Dataset&Evaluation code release (2022-10-30)

## Environment
```
conda create -n neuralmarker
conda activate neuralmarker
conda install python=3.7
pip install -r requirements.txt
```

## Dataset

We use the MegaDepth dataset that preprocessed by [CAPS](https://github.com/qianqianwang68/caps), which is provided in this [link](https://drive.google.com/file/d/1-o4TRLx6qm8ehQevV7nExmVJXfMxj657/view?usp=sharing).
We generate FlyingMarkers training set online. To genenerate FlyingMarkers validation set and test set, please execute:
```
python synthesis_datasets.py --root ./data/MegaDepth_CAPS/ --csv ./data/synthesis_validate_release.csv --save_dir ./data/flyingmarkers/validation
python synthesis_datasets.py --root ./data/MegaDepth_CAPS/ --csv ./data/synthesis_validate_short.csv --save_dir ./data/validation/synthesis
python synthesis_datasets.py --root ./data/MegaDepth_CAPS/ --csv ./data/synthesis_test_release.csv --save_dir ./data/flyingmarkers/test
```


## Training
We train our model on 6 V100 with batch size 2.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py
```

## Evaluation
organize evaluation dataset as:
```
- assets
    - D        
    - V
    - L
    - marker
```
then run
```
bash eval.sh
```

## Demo
for video demo, run
```
bash demo_video.sh
```

for harsh lighting demo with NIID-Net, run
```
bash demo_harsh_lighting.sh
```

## Acknowledgements
We thank Yijin Li, Rensen Xu, and Jundan Luo for their help.
We refer DGC-Net to generate synthetic image pairs.
