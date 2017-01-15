# Sentiment Analysis Using TensorFlow Framework

# Abstract:
  Sentiment analysis on Twitter data using Long short term Memory, a type of ## Recurrent Neural Network ## and word embedding to group words that are semantically similar. This paper has found its motivation from sentiment analysis on twitter data using POS-specific prior polarity features, using tree kernel to obviate the need for tedious feature engineering and tree kernel. The tree kernel performs approximately at the same level, both outperforming the state-of-the-art baseline
  
# Introduction:
  Microblogging websites like Friend Feed, Tumblr, Plurk, Twitter, Pinterest has become a source of many various information. Microblogs allow users to exchange small elements of content as short sentences, individual images, or links to videos. People post about their real time opinions on a lot of topics happening around the world. This could also be
reviewing a product at amazon or a movie on imDB. Some of the companies start their poll on a product through a microblog site to get a general idea on their product.
Companies study on the user’s opinion and summarize on a general sentiment. A technology to produce an overall sentiment is a challenging task. This paper, we concentrate on the Twitter blogging site.
Twitter is an online site which services news and social network where users post and read short messages called as” Tweets”.
This paper sought motivation from the paper that does sentiment analysis on twitter and classifies sentiment as “Positive”, “Negative” and “Neutral”. They built models for two classification tasks.
One is, a binary task of classifying a sentiment to positive and negative and 3 classifications such as positive, negative and neutral. They experimented using the models
  1)	Unigram Model
  2)	Feature based model
  3)	Tree kernel based
  Feature based models use previous features and proposed new features. A tree based
representation for tree kernel based model and unigram model proves to be the hardest baseline for both types of classification tasks. A combination of unigrams with features, features with tree kernels were also used.
In our paper, we propose to use Long short term Memory Model in place of all the above models with word embedding. This LSTM is a type of RNN.
