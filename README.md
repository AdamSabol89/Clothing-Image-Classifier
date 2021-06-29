# Clothing-Image-Classifier: Project Overview
- Utilized various machine learning models to classify images of clothing from 15 different possible classes with a best model validation accuracy of 73%. 
- Data was sourced from pre-existing [kaggle dataset.](https://www.kaggle.com/agrigorev/clothing-dataset-full) Data was cleaned and preprocessed resulting in a balanced dataset of 1400 observations. 
- Utilized a Convolutional Neural Network, XGBoost, and pre-trained imagent VGG16 to obtain best model. 

## Code and Resources Used:
- Python version 3.7
- Packages: pandas, numpy, sklearn, pickle, matplotlib, seaborn, cv2, tensorflow, keras.
- [Video](https://www.youtube.com/watch?v=9GzfUzJeyi0&) from DigitalSreeni on feature extraction with Conv-nets.

### Data Cleaning 
Data was sourced from a kaggle dataset with originally over 5000 images of clothing. A number of steps are required to get this data into the form needed for modeling.
- Removed all children's clothing. 
- Removed all classes with observations of less than ~100.
- Extracted data with variable "Not Sure" to classify with model after training. 
- Undersampled data to provide balanced dataset for Neural-Network classification.
- Loaded in color images and resized images with cv2.
- Pre-processed and normalized image-data.
![](https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/data_comparison.png)

### Modeling 
Now that data has been cleaned and preprocess we can begin the process of modeling, however our data-sourcing and cleaning processes have left us with a few hurdles to overcome. First, our dataset is relatively small with a balanced training set of ~1200 observations and an unbalanced training set of ~4100. However, traditional deep learning approaches work best with observations in the 100,000's. Second, Neural Network's do not work well with heavily unbalanced datasets as they can easily minimize loss by simply predicting classes with the highest number of observations. We can demonstrate this by beginning our modeling with a traditional Convolutional Neural Network. Our model consists of three convolutional layers of size 32, with maxpooling and batch normalization, then two more convolutional layers of size 64, with maxpooling and batch normalization. Finally, there are three dense layers of size 128, 256, 128, with RELU activation functions. We run the model for fifty epochs and graphs of the loss and accuracy history are provided.

![](https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/Figure_4.png) ![](https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/Figure_5.png)

As the graphs show we can only reach roughly 50% validation accuracy even after 50 epochs of training. Those familiar with neural-network training will also recognize a significant problem here. Even though our training accuracy continues to increase and loss continues to decrease our test accuracy and loss level out after about 10-15 epochs. This is typically a sign of overfitting, which is likely happening here, however there is also the lingering issue of simply not enough data in our training set. We can further explore our results by evaluating the confusion matrix. 

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/Conv_Net_CM.png">
</p>

We can see that though our model produces significant errors it is not biased against any particular classification, as mentioned earlier this is because we balanced our dataset during pre-processing. We can further evaluate our model's performance by plotting some of its predictions. Below is a plot of randomly selected images along with their actual classification and neural network prediction.

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/Conv_net_Preds.png">
</p>
Finally, we can evaluate how our model is doing by looking at how our model predicts observations which were not classified or those with the classification "Not Sure." There is no objective metric here since we do not have a proper test label, but we can get a rough estimation of how our model performs compared to human beings. 

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/Conv_Net_ns_preds.png">
</p>

### Modeling with XGBoost and Feature Extraction
I mentioned in the modeling section two main problems with our dataset. One, that we have a low-number of observations, making it not well suited for deep-learning style methods, and two our dataset is unbalanced. Both of these problems can be addressed with the use of traditional machine learning methods, i.e. tree-based methods. XGBoost is a relatively recent popular expansion of gradient boosting tree-based methods and seems to be of excellent use here. However, we can do even better than just XGBoost, by combining it with the convolutional layers of VGG16, a conv neural net trained on the ImageNet dataset, we can extract higher level features from our dataset and use them as the features in our trees. To see how this is implemented check out either the code in the XGBoost script provided in this repository or the video listed in the resources section. We will create a model with both our balanced and unbalanced datasets and see how they perform. We can evaluate the models by taking a look at their confusion matrices.

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/XGBoost_CM_Balanced.png">
</p>
<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/XGBoost_CM_full.png">
</p>

We can see from our confusion matrices that though the full-data-set makes more accurate predictions, the balanced-data-set is less susceptible to bias. This is an interesting problem similar to the bias-variance trade off which has appeared in our data. For now let's continue our analysis with the full-data-set and plot our images with their corresponding predictions. 

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/xgboost_predictions.png">
</p>

Finally, let's look at how our model predicts "Not Sure" observations. 

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/xgboost_ns_predictions.png">
</p>

### Model Comparison 
Finally we can look at how the models performed with a graph showing their various accuracy rates. We clearly see that our XGBoost model with the full data set and feature extraction has the best performance. 

<p align="center">
  <img src="https://github.com/AdamSabol89/Clothing-Image-Classifier/blob/main/figures/Figure_12.png">
</p>

