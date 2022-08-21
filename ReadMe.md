Project Description:
=====================
This project is the implementation of the social NLP paper @Naacl 2022 "A Comparative Study on Word Embeddings and Social NLP Tasks".

How to use:
==============
To use our code and reproduce our results, you have to follow these steps:

1- Download the data
=======================
The Data used in this project are:

1- Twitter-sexism and Twitter-racism datasets from "Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter"

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

Cite
====
@inproceedings{elsafoury-etal-2022-comparative,
    title = "A Comparative Study on Word Embeddings and Social {NLP} Tasks",
    author = "Elsafoury, Fatma  and
      Wilson, Steven R.  and
      Ramzan, Naeem",
    booktitle = "Proceedings of the Tenth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.socialnlp-1.5",
    doi = "10.18653/v1/2022.socialnlp-1.5",
    pages = "55--64",
    abstract = "In recent years, gray social media platforms, those with a loose moderation policy on cyberbullying, have been attracting more users. Recently, data collected from these types of platforms have been used to pre-train word embeddings (social-media-based), yet these word embeddings have not been investigated for social NLP related tasks. In this paper, we carried out a comparative study between social-media-based and non-social-media-based word embeddings on two social NLP tasks: Detecting cyberbullying and Measuring social bias. Our results show that using social-media-based word embeddings as input features, rather than non-social-media-based embeddings, leads to better cyberbullying detection performance. We also show that some word embeddings are more useful than others for categorizing offensive words. However, we do not find strong evidence that certain word embeddings will necessarily work best when identifying certain categories of cyberbullying within our datasets. Finally, We show even though most of the state-of-the-art bias metrics ranked social-media-based word embeddings as the most socially biased, these results remain inconclusive and further research is required.",
}
