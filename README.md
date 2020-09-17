# Multivariate-Multistep-Wind-Speed-Prediction

This respiratory contains the implementation codes of of both single-step and multi-step wind speed prediction using multivariate input data. The idea was to used exogenous parameters (e.g. temperature, humidity, pressure, ... , etc) to predict the wind speed for different horizons (steps ahead) without involving the history data of wind speed itself. Two main approaches were used including various deep learning transfer learning methods and the conventional neural networks models. 

## Database
The wind database were obtained from the [National Renewable Energy Laboratory’s National Wind Technology Center (NWTC)](https://midcdmz.nrel.gov/apps/sitehome.pl?site=NWTC)  of M2 tower. The M2 tower data were taken every two seconds and averaged over one minute measured at different heights (from 2 to 80 m). However, for the prediction purposes, we downsample the data to 10 minutes (averaging). The processed data can be accessed [here](https://drive.google.com/drive/folders/1x2zequLN8jUWAuyUL3Tf0LLK7u5VwStN?usp=sharing).
The data partitioned such that one year (2017) is used for training and validation (80% and 20%), and another year (2018) for testing purpose. 

## Transfer Learning
The [Keras](https://keras.io/api/applications/) pretrained deep learning models were used for this experiment. The codes were implemented in [Google-Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true). 

## Neural Network
Several neural network methods that have been proposed in the literature were implemented in [MATLAB](https://www.mathworks.com/) , feed-forward neural network (FFNN), time delay neural network (TDNN), nonlinear autoregressive exogenous model (NARX).

