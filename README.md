# Breast Ultrasound using Vision Mamba

###### Vision Mamba for Classification of Breast Ultrasound Images, MICCAI 2024

**Abstract,**  *Mamba-based models, VMamba and Vim, are a recent family of vision encoders that offer promising performance improvements in many computer vision tasks. This paper compares Mamba-based models with traditional Convolutional Neural Networks (CNNs) and  Vision Transformers (ViTs) using the breast ultrasound BUSI and B datasets. Our evaluation, which includes multiple runs of experiments and statistical significance analysis, demonstrates that some of the Mamba-based architectures often outperform CNN and ViT models with statistically significant results. For example, in the B dataset, the best Mamba-based models have a 2.61\% average AUC and a 9.17\% average Accuracy improvement compared to the best non-Mamba-based model. These Mamba-based models effectively capture long-range dependencies while maintaining inductive biases, making them suitable for applications with limited data.*
 
[[`arXiv`](https://arxiv.org/abs/2407.03552)] | [[`Cite`]](#citation) 


## Dataset

### Dataset Source
We use [BUSI dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) and [Breast Ultrasound B dataset](https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php) for this study. 

### Data Structure

The dataset should be structured as follows. We remove the mask files. The combined dataset is combined between benign and malignant classes from the two datasets.  
```bash
dataset/
	├── B
            ├── benign
              ├── 000001.png
              ├── 000002.png
              ├── ...
            ├── malignant
              ├── 000018.png
              ├── 000023.png
              ├── ...
	├── BUSI
            ├── benign
              ├── benign (1).png
              ├── benign (2).png
              ├── ...
            ├── malignant
              ├── malignant (1).png
              ├── malignant (2).png
              ├── ...
            ├── normal
              ├── normal (1).png
              ├── normal (2).png
              ├── ...
	├── Combined
            ├── benign
              ├── 000001.png
              ├── 000002.png
              ├── benign (1).png
              ├── benign (2).png
              ├── ...
            ├── malignant
              ├── 000018.png
              ├── 000023.png
              ├── malignant (1).png
              ├── malignant (2).png
              ├── ...
            ├── normal
              ├── normal (1).png
              ├── normal (2).png
              ├── ...
```

## Installation
We use Python 3.10, torch 2.1.2 with cuda 12.2 on a single A100-40GB GPU. 

Use the following command to install the required packages:
```
cd ../mamba-1p1p1
pip install -e .
pip install causal_conv1d==1.1.0
pip install timm
pip install wandb
```
## Training
Use the following commands to run the training script for each dataset:
```
CUDA_VISIBLE_DEVICES=0 bash train_script_B.sh 2>&1 | tee log_file_B.txt
CUDA_VISIBLE_DEVICES=0 bash train_script_BUSI.sh 2>&1 | tee log_file_BUSI.txt
CUDA_VISIBLE_DEVICES=0 bash train_script_Combined.sh 2>&1 | tee log_file_Combined.txt
```

If you don't want to use WandB, use --disable_wandb in each of  the train_scripts: 

```
python3 main.py --epochs $EPOCHS --data-path $DATA_PATH --output_dir $OUTPUT_DIR --arch $arch --k-folds $KFOLDS --disable_wandb
```

## Evaluation
You can use the ***analysis.ipynb*** to summarize different models performances. 

For statistical significance analysis, you can use the ***stats_pvalue.ipynb***.


## Citation
If you find this repository useful, please consider giving a star and citation (arxiv preprint):
```
@misc{nasirisarvi2024visionmambaclassificationbreast,
      title={Vision Mamba for Classification of Breast Ultrasound Images}, 
      author={Ali Nasiri-Sarvi and Mahdi S. Hosseini and Hassan Rivaz},
      year={2024},
      eprint={2407.03552},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.03552}, 
}
```