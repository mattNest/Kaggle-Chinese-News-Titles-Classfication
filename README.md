# Kaggle-Chinese-News-Titles-Classfication

This is a repository for NCTU ECM DL course final competition.
<br> Our goal is to correctly classify 10 different categories for Chinese news titles.
<br> Please refernece this page for more info about the competition: https://www.kaggle.com/c/dl-course-final-competition/overview

### Data Preprocessing 
We observed that there is an imbalance data problem for this data set based on the following histogram of labels.
Thus, we simply do a data augmentation by adjusting all the training data to the same scale. We augmented the data to
40,000 per label.
![img of label](https://i.imgur.com/6OImUP4.png)

<br> Training data is formed of "title" and "keyword". 
Firstly, drop the comma for the titles and the keywords. Secondly, we combine the title and the keyword for each ID. 
Notice that we also do the same processing for the testing data! This is because we plan to include the testing data's sentences and keywords features later in the embedding layer. Even though the testing data doesn't have labels, we can still leverage from them by including them into the embedding layer for the model.
