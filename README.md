# DeepFAN
DeepFAN was a well-designed deep learning model to discriminate malignant nodules from benign ones utilizing chest CT scans.
## Introduction
This repository contains the main source code of our benign and malignant model (DeepFAN).
### Prerequisites
- Ubuntu 16.04.4 LTS
- Python 3.6.13
- Pytorch 1.10.0+cu113
- NVIDIA GPU + CUDA_10.1 CuDNN_8.2
This repository has been tested on NVIDIA TITANXP. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

### Installation
Install dependencies: The requirements.txt file stores the installation packages that the model depends on. Please install as follows. The installation time should not exceed two hours under normal circumstances.

```
pip install -r requirements.txt
```
To mitigate potential runtime errors caused by dependency conflicts between packages, we recommend the following optimized installation procedure. This method ensures compatibility and significantly improves reliability. The standard installation approach using pip install -r requirements.txt may fail due to complex dependency requirements between packages. To address this, we've developed an enhanced installation methodology that:
#### Optimized Installation Guide for DeepFAN
1. Create Conda Environment
```
conda create -n deepfan python=3.6.13 -y
conda activate deepfan
```
2. Upgrade Pip and Install System Dependencies
```
pip install --upgrade pip
apt update && apt install -y libxml2 libgl1-mesa-glx  # Essential libraries
```
3. Install CUDA Toolkit
```
conda install -c conda-forge cudatoolkit=11.3 -y  # Use conda-forge for better compatibility
```
4. Install PyTorch with CUDA 11.3
```
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
5. Install others
```
python pip.py
```


### Usage
This article mainly introduces the training and reasoning of the main framework; other individual modules can be adjusted according to the parameters of the paper.
- deepFan model：All models are saved in the model folder. Fusion integrates the ViT module, Fine_Grained module and GCN module. The specific training and reasoning are as follows.
    ```
    # train 
    python main_dfl_2scale.py config.py
    ```
    During inference, you need to modify the ‘inference_mode’ in config.py to ‘Ture’ and "resume" is set to the file path where you saved the training model:
    ```
    # inference 
    python main_dfl_2scale.py config.py
    ```
    #### The main parameters are as following:
    - --config: Configuration file path, the configuration file path here is: ./config.py
    #### configuration file(config.py):
    - train task: config.py
      - inference_mode = False
      - model_name = 'fusion'
      - train_set_dir: image name and label csv file path of training data, please refer to the format of './sample_data/deepfan_test.csv' file.
      - val_set_dirs: image name and label csv file path of validation data, the format is the same as 'train_set_dir'. Note that during inference, the labels here do not participate in the model update calculation, but are only for easier storage and statistics.
      - mode_save_base_dir: Model output address
    - infer task: config.py
      - inference_mode = Ture
      - model_name = 'fusion'
      - resume = Save the trained model path
      - save_csv: Result output(For details on the output format, see the ./sample_data/result.csv file.)
### Sample Data
Each CT data (nii format) was cropped into 128×128×128 patches centered on the nodule and placed in the sample_data/patch_nii folder. Here we upload chest CT images of 5 cases from our dataset for demo purpose. Detailed data preprocessing details can be found in the submitted manuscript. Please note that the Sample Data is only provided to allow users to verify the workflow of the provided codes. Since the model weights are a key component of the DeepFAN model, which was derived from a commercial software, we are unable to disclose the specific values of the model weights at the moment. Users can train the model using their own datasets to get their own model weights. Interested researchers can submit a formal request via corresponding author’s email (Y.Y.), and we will consider granting access on a case-by-case basis following a qualification review.

    # ./sample_data/deepfan_test.csv file content：
    
| mask_img | label_mb | label_lb | label_sp | label_dy |
| :----- | :-----: | -----: | -----: | -----: |
| ./sample_data/nii_patch/case_1.nii.gz | 0 | 0 | 1 | 1 |
| ... | ... |  ... |  ... |  ... |

The first column (mask_img) here stores the path of the cut patch (which is also the image path of the model input), the second column (label_mb) indicates the benign or malignant label (0/benign, 1/malignant), the third column (label_lb) indicates whether the lung nodule is lobulation (0/no, 1/yes), the fourth column (label_sp) indicates whether the lung nodule has Spiculation (0/no, 1/yes), and the fifth column (label_dy) indicates the density type label (0/Ground-glass, 1/Part-solid, 2/solid).

    # ./sample_data/result.csv file content：

| p_1 | p_2 | label | file_name |
| :----- | :-----: | -----: | -----: |
| 0.052129209 | 0.947870791 | 1 | ./sample_data/nii_patch/case_5.nii.gz |
| ... | ... |  ... |  ... |

Here, the first column (p_1) stores the probability that the model outputs benign results, the second column (p_2) stores the probability that the model outputs malignant results (p_1+p_2=1), and the third column (label) stores the benign and malignant labels of the case. Note: the labels of the input files are stored here. The fourth column (file_name) stores the file path of the patch of the case. Note: the file path of the input file is stored here.

### Reproduction Notes and Important Considerations
1.Data Volume: The training process requires as much data as possible. In our experiments, we used over 10,000 CT cases. We have not tested the model with smaller datasets, but in theory, more data will generally lead to better model performance.

2.Image Preprocessing and Data Augmentation: The preprocessing and augmentation methods used in our study are described in the original paper. We did not experiment with additional or alternative augmentation techniques. The effectiveness of different preprocessing or augmentation strategies should be validated through further experiments.

3.3D Model Input: During training, we found that 3D input models are better at capturing contextual information and consistently outperform 2D models. As a result, we abandoned 2D models at an early stage and recommend using 3D models for this task.

4.Context Information: Our training experience also indicates that providing sufficient perinodular (xy-plane) context around the nodule significantly improves model performance. Ensure that the input includes enough surrounding tissue information in the xy direction for optimal results.

### Demo
The size of the five cases we give is 128×128×128, and the format is nii, which can be viewed using itk-snap. After inputting into DeepFAN, the predicted probability is output, where the label corresponding to each case is in the ./sample_data/deepfan_test.csv file, and the output probability is in the ./sample_data/result.csv file. The average inference time for each case is about 0.5s.


   

    
      
