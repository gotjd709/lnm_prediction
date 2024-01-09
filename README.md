# lnm-prediction

Implementation of lnm-prediction: Interpretable deep learning model to predict lymph node metastasis in early gastric cancer using whole slide images

![img1](./img/main.png)

This git contains five directories: `1_wsi_processing/`, `2_deep_learning/`, `3_feature_extraction/`, `4_machine learning/`, `5_eda_visualize/`

The directory [1_wsi_processing](https://github.com/gotjd709/lnm-prediction/tree/master/1_wsi_processing) contains the source code of a patch generation and Whole Slide Image(WSI) inference.

The directory [2_deep_learning](https://github.com/gotjd709/lnm-prediction/tree/master/2_deep_learning) contains the source code of tumor semantic segmentation training.

The directory [3_feature_extraction](https://github.com/gotjd709/lnm-prediction/tree/master/3_feature_extraction) contains the source code of morphology feature extraction from tumor segmentation mask.

The directory [4_machine learning](https://github.com/gotjd709/lnm-prediction/tree/master/4_machine_learning) contains the source code of machine learning for predict lymphoma node metastasis.

The directory [5_eda_visualize](https://github.com/gotjd709/lnm-prediction/tree/master/5_eda_visualize) contains the source code of exploratory data analysis(EDA) and visualization of the results.
