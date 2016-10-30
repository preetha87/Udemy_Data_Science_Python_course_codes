
# coding: utf-8

# In[1]:

import pandas


# In[2]:

#use pd.read_csv
#indicate that it is a tab separated file by using '\t' and name the columns as labels and message
messages = pandas.read_csv('SMSSpamCollection', sep='\t',
names=["label", "message"])
messages.head()


# In[3]:

#Split your data
#From scikit-learn's cross validation library, import train_test_split
#The \ symbol here means you are starting a new line
#I opted for a train-test-split of 70%-30%
from sklearn.cross_validation import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


# In[4]:

#Exploratory Data Analysis - the entire data set
#Let's check out some of the stats with some plots and the built-in methods in pandas!
messages.describe()


# In[5]:

#Use groupby to describe by label
#Grouping by the label column and then using the describe method
#as the aggregate function on the groupby method - for groupby, we always need to have an aggregate function
messages.groupby('label').describe()


# In[6]:

#As we continue our analysis we want to start thinking about the features we are going to be using.
#This goes along with the general idea of feature engineering.
#The better your domain knowledge on the data, the better your ability to engineer more features from it.
#Feature engineering is a very large part of spam detection in general.
#One feature to consider here is message length
messages['length'] = messages['message'].apply(len)
#I am only looking to apply the length function on the message column of the messages data frame -
#I am not calling it - so no need for the extra parantheses in the len function - len()
messages.head()


# In[7]:

#Let's do some data visualization based on message length
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[8]:

#Histogram of the length of the messages
#Call plot on the column of the pandas data frame
#plot is the more general method -
#set the number of bins - either 50 or 100 (just play around and see what works better)
messages['length'].plot(bins=50,kind='hist')
#Question: why does the x-axis on the histogram plot - go all theway up to 1000?
#Is there a really long message?


# In[9]:

#Find out through min, max and mean
messages['length'].max()


# In[10]:

messages['length'].min()


# In[11]:

messages['length'].mean()


# In[12]:

#Take a look at that huge message of 910 characters
messages[messages['length']==910]


# In[13]:

#Take a look at the entire message
#The iloc[0] command is just to print it out
messages[messages['length']==910]['message'].iloc[0]


# In[14]:

#Take a look at the tiny message of 2 characters
messages[messages['length']==2]


# In[17]:

#Is message length a distinguishing factor between ham and spam?
#Generate two histograms - one for ham and one for spam
#We are plotting two histograms for the column length - grouped by the factor label
messages.hist(column='length', by='label', bins=50,figsize=(10,4))
#Again, take note of the x-axis - for ham, it ranges from 0 to 1000
#while for spam, the x-axis ranges from 0 to 250
#Very interesting! Through just basic EDA we've been able to discover a trend that spam messages tend to have more characters.


# In[18]:

#The messages in the data set are in the form of strings in text format - 
#the classification algorithms need some sort of numerical feature vector in order to perform the classification task
#The task here is to convert a corpus into a vector format corpus is a word for a group of texts
#Let's use the bag of words approach - where each unique word in the text has a unique number
#we'll massage the raw messages (sequence of characters) into vectors (sequences of numbers).
#As a first step, let's write a function that will split a message into its individual words
#and return a list. We'll also remove very common words, ('the', 'a', etc..).
#To do this we will take advantage of the NLTK library. It's pretty much the standard library
#in Python for processing text and has a lot of useful features. We'll only use some of the basic ones here.
#First removing punctuation. We can just take advantage of Python's built-in string
#library to get a quick list of all the possible punctuation
import string


# In[19]:

#Let's create a function that will process the string in the message column, then we can
#just use apply() in pandas do process all the text in the DataFrame.
#Our main goal is to create a function that will process a string in the message column
#so as to get Pandas to do all the processing of the words!
#Take these raw messages and then turn them into vectors -
#take a sequence of characters and turn them into a sequence of numbers
#Create a variable called mess
mess = 'Sample message! Notice: it has punctuation.'
#First, remove the puncuation
#Instead of applying everything onto the data frame, we just mess around with the variable mess
#We can gather all the different transformations made on this and set it up as a function later on
#The following generates a string that contains puncuation marks
string.punctuation


# In[21]:

#Check characters to see if they are in punctuation
#nonpunc is a vector that is to contain all the non-puncuation marks in the
#form of capital letters and small letters - it replicates the message in mess
#(in the form strings) without the puncuation
nopunc = [char for char in mess if char not in string.punctuation]


# In[22]:

nopunc


# In[23]:

# Join the strings again to form a large string.
nopunc = ''.join(nopunc)


# In[24]:

nopunc


# In[25]:

#Next task is to remove stop words
#Stopwords are common English words
#NLTK has the most support for English language
from nltk.corpus import stopwords


# In[26]:

#Let's see a preview of the object stopwords
#in terms of the words that it contains
#Grab the 1st 10 words in stopwords
stopwords.words('english')[0:10]
#These stopwords are so common that they are not going to give us much information during classification
#Stopwords have an equally possible chance of occuring in a ham message in comparison to a spam message


# In[27]:

#split nopunc into separate words
nopunc.split()


# In[28]:

#Get rid of the stop words
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#An object with the clean message (a message without the stop words)
#examine each word in the list called nopunc.split
#consider the lower case instance of each word in nopunc.split
#if the word is not in the list of stop words, then include it in
#the clean_mess list
clean_mess


# In[31]:

#Combine all these steps of text preprocessing and put them into a function
def text_process(mess):
#Takes in a string of text, then performs the following:
#1. Remove all punctuation
#2. Remove all stopwords
#3. Returns a list of the cleaned text
# Check characters to see if they are in punctuation
 nopunc = [char for char in mess if char not in string.punctuation]
# Join the characters again to form the string.
 nopunc = ''.join(nopunc)
# Now just remove any stopwords
 return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[32]:

#Here is the original DataFrame again:
messages.head()


# In[33]:

#Need to tokenize the messages in the data frame
#It is the process of converting normal text strings
#into a list of tokens -which are the words that we actually want
#Essentially, all we are doing is to apply the function text_process
#to the column message in the messages dataframe
#The code below is just to make sure that the function is working!
messages['message'].head(5).apply(text_process)


# In[34]:

#Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to convert
#each of those messages into a vector the SciKit Learn's algorithm models can work with.
#Now we'll convert each message, represented as a list of tokens (lemmas) above,
#into a vector that machine learning models can understand.
#We'll do that in three steps using the bag-of-words model:
#Count how many times does a word occur in each message (Known as term frequency)
#Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
#Normalize the vectors to unit length, to abstract from the original text length (L2 norm)


# In[35]:

#Let's begin the first step:
#Each vector will have as many dimensions as there are unique words in the SMS corpus.
#We will first use SciKit Learn's CountVectorizer.
#This model will convert a collection of text documents to a matrix of token counts.
#We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (each row represents every word
#available in the corpus) and the other dimension are the actual documents, in this case one column per text message.
#Since there are so many messages, we can expect a lot of zero counts for thepresence of that word in that document.
#Because of this, SciKit Learn will output a Sparse Matrix. A sparse matrix is a matrix where most of the values are 0.
from sklearn.feature_extraction.text import CountVectorizer


# In[36]:

#There are a lot of arguments and parameters that can be passed to the CountVectorizer.
#In this case we will just specify the analyzer argument in order to set it to be our own previously defined function
#which is text_process
#We define an object called bag_of_words_transformer
bag_of_words_transformer = CountVectorizer(analyzer=text_process)


# In[37]:

#Next fit the bag_of_words model to that column called message in the messages dataframe
bag_of_words_transformer.fit(messages['message'])
print len(bag_of_words_transformer.vocabulary_)
#The warning below is because of some weird unicode in the text message such as the pound symbol


# In[38]:

message4 = messages['message'][3]
print message4


# In[39]:

#See if it works!
bow4 = bag_of_words_transformer.transform([message4])
print bow4
print bow4.shape
#Numbers like 4068, 4629 - stand for word number 4068 - word number 4629
#It looks like message 4 has 7 unique words (after removing the common stop words) - two of these appear twice


# In[40]:

#This means that there are seven unique words in message number 4 (after removing common stop words).
#Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:
print bag_of_words_transformer.get_feature_names()[4068]


# In[41]:

print bag_of_words_transformer.get_feature_names()[9554]


# In[42]:

#Use dot transform on our bag of words Now we can use dot transform on our Bag-of-Words object
#and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts
#for the entire SMS corpus is a large, sparse matrix:
messages_bow = bag_of_words_transformer.transform(messages['message'])


# In[43]:

print 'Shape of Sparse Matrix: ', messages_bow.shape
print 'Amount of Non-Zero occurences: ', messages_bow.nnz
#nnz is part of scipy.sparse.dia_matrix.nnz
# outputs the number of nonzero values
#explicit zero values are included in this number
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0]
* messages_bow.shape[1]))


# In[44]:

#TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used
#in information retrieval and text mining. This weight is a statistical measure used to evaluate
#how important a word is to a document in a collection or corpus. The importance increases
#proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.
#Why do we need this?
#If we are trying to compare a text message of 100 characters and you have a few others that are text messages of 900 characters
#You will need to divide the number of times term t appears in a document by the Total number of terms in the document
#otherwise, you'll begin to skew your weights for common terms
#To expand that idea, we have:
#IDF: Inverse Document Frequency, which measures how important a term is.
#While computing TF, all terms are considered equally important.
#However it is known that certain terms, such as "is", "of", and "that",
#may appear a lot of times but have little importance. Thus we need to weigh down the frequent
#terms while scale up the rare ones, by computing the following:
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
#Example:
#Consider a document containing 100 words wherein the word cat appears 3 times.
#The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03.
#Now, assume we have 10 million documents and the word cat appears
#in one thousand of these. Then, the inverse document frequency (i.e., idf)
#is calculated as log(10,000,000 / 1,000) = 4.
#Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.


# In[45]:

#Let's go ahead and see how we can do this in SciKit Learn:
#Import from scikit-learn's feature extraction library, import the TfidTransformer
from sklearn.feature_extraction.text import TfidfTransformer


# In[46]:

#Create an object called tfid_transformer and set it equal to TfidTransformer for which
#you need to call the fit method and pass the fit mehod on to that bag of words object called messages_bow
tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[47]:

#We next, need to transform the bag-of-words counts as a vector
#Recall that messages_bow is a sparse matrix - this sparse matrix was generated by the bag of words transformer
##This is essentially a 2 dimensional matrix
#The entire sparse matrix needs to transformed by the
#tfidf_transformer - into a series containing all the unique words (in the form of numbers for each unique word)
#and the number of times these unique words appear in teach message in the data frame
#To transform the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape


# In[48]:

#Take a look at how the tfidf transformer words for a single message
#Recall that we applied the bag of words transformer to a single message
#and set this equal to bow4
#Let's apply the TFIDF transformer to bow4
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4


# In[49]:

#Recall that we earlier split our original data set into the training set and the test set in the proportion of 70:30
#print the size of the training set, testing set and the size of the entire data set

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
#The test size is 30% of the entire dataset (1672 messages out of total 5572)
#and the training is the rest (3900 out of 5572). Note the default split would have been 70/30


# In[50]:

#Creating a Data Pipeline
#Let's run our model again and then predict off the test set.
#We will use SciKit Learn's pipeline capabilities to store a pipline of workflow.
#This will allow us to set up all the transformations that
#we will do to the data for future use. Let's see an example of how it works:
#Pipeline is used to make our jobs easier - set up all the pre-processing and modeling steps
#into a single pipeline
from sklearn.pipeline import Pipeline


# In[52]:

#Messages are represented as vectors - we can start training a ham vs. spam classifier
#Import multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB


# In[53]:

#Now we can directly pass message text data and the pipeline will do our preprocessing for us!
#We can treat it as a model/estimator API:
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)), # strings to token integer counts
('tfidf', TfidfTransformer()), # integer counts to weighted TF-IDF scores
('classifier', MultinomialNB())]) # train on TF-IDF vectors w/ Naive Bayes classifier


# In[55]:

#We can use pipeline in such a way that we can pass fit on it and call the training set msg_train!
pipeline.fit(msg_train,label_train)
#The warning below arises as the text contains special characters like a $ sign
#and that such signs were not converted


# In[56]:

#We can use pipeline in such a way that we can pass predict on it and call the testing set test_train!
predictions=pipeline.predict(msg_test)


# In[58]:

#Import classification_report
from sklearn.metrics import classification_report


# In[59]:

#Get a classification report
#We need a classification report for our model on a true testing set!
#How well the spam/ham classifier did on the testing set - as far as predicting labels are concerned (classification task)
print classification_report(predictions,label_test)


# In[ ]:



