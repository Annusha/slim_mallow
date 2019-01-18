# Unsupervised Learning and Segmentation of Complex Activities from Video

Unofficial python implementation of the paper Unsupervised Learning and Segmentation of Complex Activities from Video. [a link](https://arxiv.org/pdf/1803.09490.pdf)

## Results and changes:
Full log for the Breakfast dataset result is here.  [a link](https://github.com/Annusha/slim_mallow/blob/master/results/log)  
Reduced log just with total accuracy and per action MoFs is here.  [a link](https://github.com/Annusha/slim_mallow/blob/master/results/evaluation.md)  
Changes in the model are here.  [a link](https://github.com/Annusha/slim_mallow/blob/master/results/changes.md)  

## Preceding steps:

### To create environment:

```conda create --name <env> --file requirements.txt```

### Data:

At this moment it's easy to run for the Breakfast dataset, for any other there should be made some changes (description for it will be added if necessary).

#### Breakfast Dataset
For each frame there should be precomputed feature vector. (originally dense trajetories).  
For each video there should be represented as one file contained matrix (row is a frame feature) which can be loaded via np.load (e.g. txt file, gz archive).   

--dataset_root=/path/to/your/folder : root for everything, there are all other subfolders located.  
--data=whatever_you_want  
then feature location is /dataset_root/feat/whatever_you_want/ascii  
All features of the same precomputed type (or folders with them, any hierarchy) should be eventually in /dataset_root/data/ascii/here_your_features  

--gt=folder_name: in folder /dataset_root/data should locate --gt=folder_name (default name groundTruth, full path: /dataset_root/data/gt) which contains ground truth labels for each your training file with the same name but without extension (txt/gz).  
If you already have mapping.txt file (correspondences between numerical indexes and subactivity names, look at data_ex/mapping.txt) put this file into /dataset_root/data/gt/mapping/mapping.txt.  
If you don't have such file and gt segmentation is in the format as for YTI (data_ex/YTI_seg_ex) you can create it from the ground truth segmentation:   
```python YTI_utils/mapping.py```  
Comment 27,28 lines if you do not need background class, gt_folder (line 21) - your folder with ground truth labels  

--feature_dim=64 : dimensionality of frame feature  
--end=txt : extension of your stored features (txt/gz/whatever)  
--data_type=4 (default) : others types are different precomputed features for breakfast.  

### Other important parameters:

--gmms=many : paper version of Gaussian Mixture Models for subactions  
--gmms=one : slim version (more than in 4 time faster)  

--ordering=True: using Generalized Mallow Model to create order (default=True)  

--full=False : quick check on the part of the videos (take first 20 videos, default=True, namely all data will be readed)  
  
--save_model=False : to save trained embedding weights  

--prefix=whatever : prefix for you data names which will be stored, namely log file, segmentation for each video (can't turn off), embedding weights (if specified).  

### How to run:

```
python pipeline.py --your_params
```

### Full set of parameters

```
usage: pipeline.py [-h] [--subaction SUBACTION] [--dataset DATASET]
                   [--data_type DATA_TYPE] [--dataset_root DATASET_ROOT]
                   [--data DATA] [--gt GT] [--feature_dim FEATURE_DIM]
                   [--ext EXT] [--seed SEED] [--lr LR] [--lr_adj LR_ADJ]
                   [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                   [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                   [--embed_dim EMBED_DIM] [--epochs EPOCHS]
                   [--gt_training GT_TRAINING] [--gmm GMM] [--gmms GMMS]
                   [--reg_cov REG_COV] [--ordering ORDERING] [--bg BG]
                   [--bg_trh BG_TRH] [--viterbi VITERBI]
                   [--save_model SAVE_MODEL] [--resume RESUME]
                   [--resume_str RESUME_STR]
                   [--save_likelihood SAVE_LIKELIHOOD]
                   [--save_embed_feat SAVE_EMBED_FEAT] [--full FULL]
                   [--zeros ZEROS] [--vis VIS] [--prefix PREFIX] [--log LOG]
                   [--log_str LOG_STR]

optional arguments:
  -h, --help            show this help message and exit
  --subaction SUBACTION
                        measure accuracy for different subactivities
  --dataset DATASET     Breakfast dataset (bf) or YouTube Instructional (yti)
  --data_type DATA_TYPE
                        for this moment valid just for Breakfast dataset0:
                        kinetics - features from the stream network1: data -
                        normalized features2: s1 - features without
                        normalization3: videos4: new features, not specified
                        earlier
  --dataset_root DATASET_ROOT
                        root folder for dataset:Breakfast / YTInstructions
  --data DATA           direct path to your data features
  --gt GT               folder with ground truth labels
  --feature_dim FEATURE_DIM
                        feature dimensionality
  --ext EXT             extension of the feature files
  --seed SEED           seed for random algorithms, everywhere
  --lr LR               initial learning rate
  --lr_adj LR_ADJ       will lr be multiplied by 0.1 in the middle
  --momentum MOMENTUM   momentum
  --weight_decay WEIGHT_DECAY
                        regularization constant for l_2 regularizer of W
  --batch_size BATCH_SIZE
                        batch size for training embedding (default: 40)
  --num_workers NUM_WORKERS
                        number of threads for dataloading
  --embed_dim EMBED_DIM
                        number of dimensions in embedded space
  --epochs EPOCHS       number of epochs for training embedding
  --gt_training GT_TRAINING
                        training embedding (rank model) either with gt labels
                        or with labels gotten from the temporal model
  --gmm GMM             number of components for gaussians
  --gmms GMMS           number of gmm for the video collection: many/one
  --reg_cov REG_COV     gaussian mixture model parameter
  --ordering ORDERING   apply Mallow model to incorporate ordering
  --bg BG               if we need to apply part for modeling background
  --bg_trh BG_TRH
  --viterbi VITERBI
  --save_model SAVE_MODEL
                        save embedding model after training
  --resume RESUME       load model for embeddings, if positive then it is
                        number of epoch which should be loaded
  --resume_str RESUME_STR
                        which model to load
  --save_likelihood SAVE_LIKELIHOOD
  --save_embed_feat SAVE_EMBED_FEAT
                        save features after embedding trained on gt
  --full FULL           check smth using only 15 videos
  --zeros ZEROS         if True there can be SIL label (beginning and end)that
                        is zeros labels relating to non actionif False then 0
                        labels are erased from ground truth labeling at all
                        (SIL label), only for Breakfast dataset
  --vis VIS             save visualisation of embeddings
  --prefix PREFIX       prefix for log file
  --log LOG             DEBUG | INFO | WARNING | ERROR | CRITICAL
  --log_str LOG_STR     unify all savings

```


