# Kaggle-Chinese-News-Titles-Classfication

This is a repository for NCTU ECM DL course final competition.
<br> Our goal is to correctly classify 10 different categories for Chinese news titles.
<br> Please refernece this page for more info about the competition: https://www.kaggle.com/c/dl-course-final-competition/overview

### Data Preprocessing 
We observed that there is an imbalance data problem for this data set based on the following histogram of labels.
Thus, we simply do a data augmentation by adjusting all the training data to the same scale. We augmented the data to
40,000 per label.
<br> ![img of label](https://i.imgur.com/6OImUP4.png)

<br> Training data is formed of "title" and "keyword". 
Firstly, drop the comma for the titles and the keywords. Secondly, we combine the title and the keyword for each ID. 
**Notice that we also do the same processing for the testing data!** This is because we plan to include the testing data's sentences and keywords features later in the embedding layer. Even though the testing data doesn't have labels, we can still leverage from them by including them into the embedding layer for the model.

<br> We use jieba, a popular library for processing Chinese, to prepare the training data we need. However, only splitting out the
words doesn't provide enough information for our models. We only pick some certain POS(詞性) for our model to learn. 


```python
def get_sent_list(sentences):
    sent_list = []
    flag_list = ['n','ng','nr','nrfg','nrt','ns','nt','nz','s','j','l','i','v','vn','eng']
    
    for num in range(len(sentences)):
        
        if num % 10000 == 0:
            print("No. %d" % (num))
        
        sent = sentences[num]
        sent = pseg.cut(sent) # get part of speech

        tmp_list = []
        for word, flag in sent:
            if flag in flag_list:
                tmp_list.append(word)

        sent_list.append(tmp_list)
    
    
    return sent_list
```
Then we use Tokenizer from keras to split out the training inputs / labels we need for our models. We reserve 20% of training
data for validation with random state = 42.

### Model Construction

#### LSTM Model 1: Own trained Embedding Layer
We experiment our first model with own trained embedding layer. There are lots of pre-trained word vectors, such as Bert and
XLnet. However, we would like to experiment with own-trained embedding layer first to see the performance. Choosing Epoch
= 10, we can get the acc = 0.93603. Notice that we also put back the 20% validation data back and trained the model again.
The acc would rise slightly for learning more information from the data.

<br> ![img of label](https://i.imgur.com/Gm54m8l.png)

#### LSTM Model 2: Pre-trained Embedding Layer
Next, we tried using pre-trained word vector for the embedding layer. The pre-trained word-vectors are from this github repo (
Mix-large). However, the acc only goes to 0.93596 with 15 epochs of training. Maybe there are more powerful pre-trained
word vectors. 
<br> The pre-trained vectors are from here: https://github.com/Embedding/Chinese-Word-Vectors

<br> ![img of label](https://i.imgur.com/c0PRfaa.png)

