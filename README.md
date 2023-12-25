Official repository for the WACV 2024 paper [Multi-view Classification with Hybrid Fusion and Mutual Distilation](https://openaccess.thecvf.com/content/WACV2024/papers/Black_Multi-View_Classification_Using_Hybrid_Fusion_and_Mutual_Distillation_WACV_2024_paper.pdf). Here, you'll find our code to train and evaluate our method, MV-HFMD. Currently, we provide code to run MV-HFMD on the Hotels-8k dataset. 

## Instructions

To train our method on Hotels-8k, first, download the dataset from this [link](https://tuprd-my.sharepoint.com/:u:/g/personal/tul03156_temple_edu/EdVGFFJyQKpGqxmk-WeApP8BLzHIaQ2XYGhhR6E1s0ntqQ?e=qR5rZf). Unzip the file into the desired directory. Then, run

    python3 main.py --data-directory {DATA_DIRECTORY}
    
You can toggle the mutual distillation loss function with the argument 
    
    --use_mutual_distillation_loss {True/False}
    
And then the number of images per collection that you wish to train and evaluate on

    --num_images {2/3/4}

By default, the model will generate classification predictions for each individual image and then the entire multi-view collection. These are given in the model output dictionary under the keys 'single' and 'mv_collection', respectivefully. 

## Requirements:

* Python 3
* torch
* numpy
* timm
* einops
    
## Citation: 

If you find our work helpful in your research, please consider citing:

    @inproceedings{black2024multi,
      title={Multi-View Classification Using Hybrid Fusion and Mutual Distillation},
      author={Black, Samuel and Souvenir, Richard},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={270--280},
      year={2024}
    }

