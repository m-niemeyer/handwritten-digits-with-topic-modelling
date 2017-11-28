# Unsupervised learning of handwritten digits with topic modelling
_Author: Michael Niemeyer 2017_

This work is an example of how an understanding of handwritten digits can be learnt unsupervised, that is without any form of labels.

## General Idea
The idea is the following: Each image consists of 28 * 28 = 784 pixels p_i with values ranging from 0 to 255 (greyscale image). We interpret each pixel p_i as a word, and its intensity value corresponds to its frequency. This way, we are able to model the image data as a collection of "documents" each consisting of 784 different "words" with different frequencies.
Please see this link of the corresponding kaggle competition to find the training and test data: https://www.kaggle.com/michaelnie/unsupervised-learning-with-topic-modelling/data

## Topic Modelling Concept
The reason I used terms like "documents" and "words" is that the topic modelling concept is best understood in the text data context. In short, it is a Bayesian learning model which assumes a generative process of the data. In the text context, it assumes that each document of a collection exhibits different proportions of "topics", and each word has probabilities associated with each topic.
For example, let's say we have 30 newspaper articles (collection), and the topics across the collection ranges from politics, to sports and to local news. We assume that each of the article (document) exhibits proportions for each topic, e.g. 80% sports, 10% politics, and 10% local news. Finally, the possible words have different probabilities for the different topics, so that "football", "nutrition", and "basketball" may have higher probabilities for the sports topic while "money", "government", and "elections" may have higher values for the politics topics.

Finally, the inference of the topic structure is performed by reversing the generative process.

In our case, the documents are the images, the words are the pixels p_i, their frequency are the intensity values, and we want to extract the underlying topic structure, while each topic corresponds to a number from 0 to 9.
It is quite remarkable that this simple approach works although we only consider the intensity values of the pixels - we completely remove spatial information!

## Further Information
If you are interested in topic modelling, the best way to start is to look at one of the creator's website: http://www.cs.columbia.edu/~blei/topicmodeling.html

Otherwise, feel free to contact me any time if you have questions, suggestions, etc.!
