# CUSP
**_Official implementation for paper [Custom Structure Preservation in Face Aging
]()_** (Accepted on ECCV '22)
 
## Preparation
### DEX model
The procedure to obtain the weights needed for the DEX age classifier
are described in [HRFAE Github](https://github.com/InterDigitalInc/HRFAE)'s README.md.

### Docker image

For improved efficiency it is recommended to use a Docker image.

```bash
docker build -t i2idocker .
```

## Data
### FFHQ Restricted Range
__FFHQ-RR__ dataset is built following its corresponding instructions available on [HRFAE: High Resolution Face Age Editing](https://arxiv.org/pdf/2005.04410.pdf).
### FFHQ Lifespan
__FFHQ-LS__ dataset is downloaded from [FFHQ Aging Dataset Github](https://github.com/royorel/FFHQ-Aging-Dataset).

## Models
### Available pretrined models

Two models have been released depending on the dataset used:
- Trained on __FFHQ-RR__ dataset: [Google Drive](https://drive.google.com/drive/folders/1ilazawzdIiNZq_jMxW-Gufae6x7SM66t?usp=sharing).
- Trained on __FFHQ-LS__ dataset: [Google Drive](https://drive.google.com/drive/folders/1C3zhHNFAXmmBAtbUoKQNP6nwJ5MHgBmw?usp=sharing).

### Train a new model
#### FFHQ-RR
```
DOCKER_WORKDIR=/workdir
USER_DIR=/<USER_PATH>/
GPU_ID=0
RELATIVE_DATASET_PATH=../datasets/ffhq_augm
docker run --gpus all --rm --user "$(id -u):$(id -g)" --workdir $DOCKER_WORKDIR -v $USER_DIR:$DOCKER_WORKDIR -e HOME=$DOCKER_WORKDIR -w $DOCKER_WORKDIR/cusp-pytorch i2idocker:latest python train.py --outdir ./training-runs_ffhqaug --gpus $GPU_ID --cfg 224 --dataset ffhq_aug --age_np $RELATIVE_DATASET_PATH/ffhq_aug_tr.npy --age_np_test $RELATIVE_DATASET_PATH/ffhq_aug_ts.npy --data $RELATIVE_DATASET_PATH/images --classifier_path ./dex_imdb_wiki.caffemodel.pt --downsamples 5 --bias False --class_w 0.1 --skip_layers 2 --skip_kind linear --age_margin 0 --rgb_attention False --rgb_reg none --soft_margin True --act_reg l2 --learn_mask none --mixing_prob 0.0 --disc_class True --cmap_kind number --fake_rec false --skip_grad_blur gb --blur_skip true --blur_msk random --cycle_w 10 --bottleneck_class false --style_enc true --class_kind max
```

#### FFHQ-LS
```
DOCKER_WORKDIR=/workdir
USER_DIR=/<USER_PATH>/
GPU_ID=0
RELATIVE_DATASET_CSV_PATH=../datasets/FFHQaging.csv
RELATIVE_DATASET_IMAGES_PATH=../datasets/FFHQaging
RESNET_FFHQ_LS_PATH=./trained_models/resnet_ffhq_ls.pt
docker run --gpus all --rm --user "$(id -u):$(id -g)" --workdir $DOCKER_WORKDIR -v $USER_DIR:$DOCKER_WORKDIR -e HOME=$DOCKER_WORKDIR -w $DOCKER_WORKDIR/cusp-pytorch i2idocker:latest python train.py --outdir ./training-runs-ffhqlat --gpus 3 --cfg 224 --dataset ffhq_lat --csv $RELATIVE_DATASET_CSV_PATH --data $RELATIVE_DATASET_IMAGES_PATH --classifier_path $RESNET_FFHQ_LS_PATH --downsamples 5 --bias False --class_w 0.1 --skip_layers 2 --skip_kind linear --act_reg l2 --skip_grad_blur gb --learn_mask none --mixing_prob 0.0 --disc_class true --fake_rec false --cycle_w 10 --blur_skip true --blur_msk random --cmap_kind number
```
### (Google Colab) Load a pretrained model

You can run the sample notebook in [Google Colab](https://colab.research.google.com/drive/1FM0u5E3M4D5zH9ru-OIQhR56vQqvkl1U?usp=sharing).

## Citation

_pending_

## License
_The base code was forked from [StyleGAN2-ADA-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)_
