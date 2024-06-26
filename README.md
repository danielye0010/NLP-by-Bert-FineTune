# Approach
The challenge is to perform natural language processing (NLP) on a number of tweets, determining the 
tweets about natural disasters.We used BERT, Bidirectional Encoder Representations from Transformers, 
for this challenge. BERT is a powerful model that is suitable for many NLP tasks, for example, text 
classification.

The data has been provided by Kaggle. All training data are stored in a single csv file. Each entry contains 
an unique identifier, content of the tweet, keywords and location associated with the tweet, and a binary 
identifier suggesting if this tweet is about a natural disaster. All testing data are stored in another csv file 
with a similar format but without the binary identifier. All data were loaded into 2 categories, training and 
testing. Training data has been further divided into 3 categories, text, keywords, and location. All three 
sub categories of the training data are stored as single data sets, with their associated binary identifier. Not 
every entry in the training dataset includes keywords or location. Therefore, those entries that do not 
contain keywords or location were pruned from the keywords and location datasets for training. The 
testing data are divided into the three same sub categories as well.


For each data set among the training data, an instance of pretrained BERT model was fine-tuned to 
achieve optimal performance (minimizing loss and maximizing training accuracy) given the input (text, 
keywords, or location). Parameters used for fine-tuning the models are explained in detail in the 
Implementation details section. Each fine-tuned model contains 5 layers: an input layer, which are the 
input text, keyword, or location, a preprocessing layer, in which text inputs are converted to numeric 
tokens as input to the BERT model, an encoding layer, which outputs the result of the BERT model, a 
dropout layer to avoid overfitting, and a dense layer to give each input a final score. Then the sigmoid 
activation function is used to determine if a tweet is disastrous given the final score. Each model was 
trained multiple times with different parameters, such as different epoch sizes and batch sizes and 
randomness to achieve the highest accuracy. To avoid unnecessarily re-training the model, each model 
was saved on the hard drive and can be loaded in future use.


With the saved find-tuned models, the testing data can be inferenced and produce a prediction on whether 
a tweet is on disasters, leveraging the output for the text, keywords, and location. For more details please 
refer to the experiment section.

# Implementation  Details
For this project, we utilized the Golab and high performance GPU runtime, which offered us ample 
resources for training out models. With 40GB of GRAM and 82.5GB of RAM, we were able to efficiently 
train our models and handle large amounts of data. Training the model took up to 17G of GRAM. In 
terms of packages, we used a variety of tools beyond the basic ones like numpy. We used TensorFlow as 
BERT models work with tensors, TensorFlow Hub, from which the BERT encoder can be imported, and 
TensorFlow Text, from which the BERT preprocessor can be imported. We also used the AdamW 
optimizer from the official.nlp library. By experimentation, we concluded that the loss is minimized and the accuracy is maximized for the model trained with tweet contents at 5 epochs. For the keywords model, 
the optimal loss and accuracy occurs at around 7 epochs. In addition, for the location model, the optimal 
loss and accuracy occurs at around 3 or 4 epochs. The batch size used was 64. Larger batch size will 
create greater memory overhead so we did not attempt to train the model with higher batch size. Amount 
of parameter tuning was limited as training each model takes too much resource and out available 
computation units are limited.

# Experiments
We mainly implemented and trained three different models for different situations. The model on text was 
trained for 5 epochs. 5 was selected by plotting both loss and accuracy against number of epochs and 
taking the number of epochs with the optimal value. Same method was used on the keywords model and 
location model.

# Discussion
At the time of the report, our result ranks at 242 among over 1000 submissions.
After evaluating the F1 scores of all predictions, we find that the general model for the ‘text’ variable has 
the best prediction performance. This is a bit out of line with our expectations, which we initially 
expected to achieve the best results in the predictions of the combined model. We postulate that taking the 
majority vote of the three models may not be the best approach for the combined model, as this implies 
that three models are weighted equally but they have different training accuracy. We propose a different 
scheme for the combined model: first storing the training accuracy of 3 models in a vector, normalize it 
and save the transpose as acc. Then save the results of the three models as a matrix of width 3. Multiply 
the matrix by acc and then iterate through the result, saving any value over 0.5 as 1 and 0 otherwise. We 
also propose that the model can be better tuned if we could train the models in the TPU since BERT may 
work better with the TPU as tensor operations are best optimized on the TPU.

# Competition Description
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:

The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified. If this is your first time working on an NLP problem, we've created a quick tutorial to get you up and running.

Tweet source: https://twitter.com/AnyOtherAnnaK/status/629195955506708480
