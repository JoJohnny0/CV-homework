# Efficient Anomaly Detection in Industrial Images using Transformers with Dynamic Tanh

This repository contains the code used to implement an efficient Anomaly Detection system based on Vision Transformers, exploring the importance of hyperparameter tuning and the recently proposed [Dynamic Tanh](https://arxiv.org/abs/2503.10622).

The starting point is [VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2104.10036), integrated with the [original repository](https://github.com/pankajmishra000/VT-ADL) from one of the authors.

## How to Run the Code

### Install the Dependencies

To install the dependencies from [requirements.txt](requirements.txt), run

```bash
pip install -r requirements.txt
```

It contains the required libraries to run the code, alongside their version used in the experiments. Other versions may work as well.

### Install libgl1

`libgl1` is required for `opencv-python` to work. To install it, use

```bash
apt-get install libgl1
```

### Download the Datasets

Follow the links [[MVTech](https://www.kaggle.com/datasets/ipythonx/mvtec-ad), [BTAD](https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection)] to download the datasets from Kaggle and save them in `datasets/{dataset_name}`. Your final folder should contain `datasets/MVTech/bottle/...`

### Run the Code

Use [main.ipynb](main.ipynb) to run the code. It contains, in the [Globals](main.ipynb#globals) section, all the parameters that can be changed in order to continue (or replicate) the experiments.
