Project Description:
=====================
This project is the implementation of the social NLP paper @Naccl 2022 "A Comparative Study on Word Embeddings and Social NLP Tasks".

How to use:
==============
To use reproduce this platform, you have to follow these steps:

1- Download the data
=======================
The Data used in this project are:

1- Twitter-sexism and Twitter-racism datasets from "Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitte "

2- Jigsaw-toxicity Dataset https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

3- Kaggle-insults daraset https://www.kaggle.com/c/detecting-insults-in-social-commentary/data

4- HateEval dataset from "SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter"

The datsets can be obtained by contacting the authhors. After hte datsets are downloaded, they should be placed in the following path "Data/Textual_Data" in the main project directory.

**After downloading the data place them in Data/Textual_data**

2- Download word embeddings
============================
In this project We use different word embeddings to train BiLSTM model. 

1- Word2vec: https://github.com/tmikolov/word2vec

2- Glove-WK and glove-Twitter can be downloaded from https://nlp.stanford.edu/projects/glove/

3- Urban Dictionary: http://smash.inf.ed.ac.uk/ud-embeddings/

4- Chan: https://textgain.com/8chan


**After that place the downloaded file in the following folder Data/word_embeddings**

3- Hurtlex Data
=================
All the Hurtlex data is provided in the following folder Data/Hurtlext_data


3- Cyberbullying Detection - Model Training
===============================================
To train the used models, you have to install all the packages in requirements.txt. Then run bash files Train_model.sh after specifying the model, dataset, word emebddings to trian. The trianing process saves the trained models in trained_models.


Disclaimer
==========
While every care has been taken to ensure the accuracy of the data and code provided in this repository, the authors, the University of the West of Scotland, and Oakland University do not provide any guaranties and disclaim all responsibility and all liability (including without limitation, liability in negligence) for all expenses, losses, damages (including indirect or consequential damage) and costs which you might incur as a result of the provided data being inaccurate or incomplete in any way and for any reason. 2021, University of the West of Scotland, Scotland, United Kingdom.


Also, put the following reference in the beginning of the readme:
=================================================================
@inproceedings{elsafoury-2022-comparative_analysis,
    title = "A Comparative Study on Word Embeddings and Social NLP Tasks",
    author = "Fatma Elsafoury, Steven R. Wilson, Naeem Ramzan",
    booktitle = "Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics: The Social NLP workshop",
    month = July,
    year = "2022",
    address = "Seattle, US"}
