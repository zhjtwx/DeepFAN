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

#### Usage
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
      - val_set_dirs: image name and label csv file path of validation data, the format is the same as 'train_set_dir'.
      - mode_save_base_dir: Model output address
    - infer task: config.py
      - inference_mode = Ture
      - model_name = fusion
      - resume = Save the trained model path
      - save_csv: Result output
### Sample Data
Each CT data (nii format) was cropped into 128×128×128 patches centered on the nodule and placed in the data/patch_nii folder. Here we upload chest CT images of 5 cases from our dataset for demo purpose. Detailed data preprocessing details can be found in the submitted manuscript. Please note that the Sample Data is only provided to allow users to verify the workflow of the provided codes. Since the model weights are a key component of the DeepFAN model, which was derived from a commercial software, we are unable to disclose the specific values of the model weights at the moment. Users can train the model using their own datasets to get their own model weights. Interested researchers can submit a formal request via corresponding author’s email (Y.Y.), and we will consider granting access on a case-by-case basis following a qualification review.
      
