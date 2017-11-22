# Sentiment_Analysis
NLP model using neural nets for sentiment analysis of Amazon reviews

## Aim
The code addresses the problem presented by the following kaggle challenge:

https://www.kaggle.com/bittlingmayer/amazonreviews

We have access to a training dataset of 3.6M (400K) exemplary Amazon reviews for training (validation).
The reviews are in raw text and they are sorted as 1/2-stars reviews or 4/5-stars reviews (3-star reviews, i.e. reviews with neutral sentiment were not included in the original).

The goal is to devise a scheme to:

1-Perform text cleaning removing noise, common stop words (english ones only are considered in this first implementation),  punctuation and words lemmatization.

2-Create a bag-of-words and implement a vector representation for each review. Although we also considered simple 1-hot representation, in this implementation the preferred representation is the pre-trained dense Google Word2Vec database, available at:

https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

3-With all the above, we implement a neural net model in tensorflow for classification (see scheme below).

## Model

![Alt text](summary/nn_schematics.png?raw=true "model for the neural net implemented. Some figures are adapted from https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/")

We explain the model realization starting from the scheme above. After noise removal, the each review is converted into a dense representation using the pre-trained Google Word2Vec model. Note that all locations where to find training/validation and other database files on the local drive should be specified using the configuration file Sentiment_Analysis/data/Config/file config_sentiment.cfg. The same file also allows to specify the neural net dimensionality and training parameters.

The dense review representation is fed into the neural net (when using Word2Vec, a default 1x300 dimensional vector is used). In the configuration file one can specify hidden layers for the net of arbitrary number and dimension.

Importantly, as 3.6M reviews represent a sizable database, the user can (in the configuration file) specify how many of these examples should be retained during training. By default (i.e. only to show the code capabilities) we retain 20'000 reviews for training and 10'000 for validation.

### Prerequisites

- Python 3.6
- [Tensorflow 1.2.1](https://pypi.python.org/pypi/tensorflow/1.2.1)
- [SciPy](http://www.scipy.org/install.html)
- [h5py](http://www.h5py.org/)
- [Progressbar](https://pypi.python.org/pypi/progressbar2/3.18.1)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [nltk](http://www.nltk.org/install.html)
- [sklearn](http://scikit-learn.org/stable/install.html)
- [string](https://docs.python.org/3/library/string.html)
- [bz2](https://docs.python.org/2/library/bz2.html)
- [gensim](https://pypi.python.org/pypi/gensim)
- [urllib](https://docs.python.org/3/library/urllib.html)

## Use

First, do:
```
git clone https://github.com/fcasola/Sentiment_Analysis 
```
In order to perform a full training of the neural net, __after making sure of having all datasets available__, type:
```
cd Sentiment_Analysis/model_nn/
python NLP_training.py
```
At the moment, this Git version contains a pre-trained model, highly NOT optimized and featuring no hidden layers. This geometry makes the model similar to a standard logistic regression, to which (sklearn one) it is compared at the end of the training (i.e. the accuracy on the validation set is printed out to screen).
The training will also save the loss vs epoch time under the training_Loss.h5 file in the data/model folder.

By default, a model is available in the data/model folder, so an interactive console can be launched __without need of training__ by typing:
```
cd Sentiment_Analysis/model_nn/
python NLP_prediction.py
```
The code will load the trained model and a smaller version of the Word2Vec database (only the part used during training).
As the screenshot below shows, the program will now ask to type a review and will try to judge the quality of the product that you bought:

![Alt text](summary/example_console.png?raw=true "Example of how the console app works.")
