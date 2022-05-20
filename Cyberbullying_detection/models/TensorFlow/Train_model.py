'''
This python file trains a new model,set by the user, on a new dataset, set by the user, and saves the model's
perfromance evaluation as a text file in the path determined by the user, and the trained model
in the path determined by the usrs. Please refer back to the documentation to learn more about the parameters set by the users
'''
import sys
import pandas as pd
import models_training,helpers, embeddings
import argparse

# the function being called when the python file is executed
def main():
    # set the parameters to be sent by the users in the bash file with their properties.
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_ds_file_path", default=None, type=str, required=True,
                        help=" the path of the training dataset .csv file in the Data folder")
    parser.add_argument("--test_ds_file_path", default=None, type=str, required=True,
                        help="the path of the test dataset .csv file in the Data folder")
    parser.add_argument("--text_col_name", default=None, type=str, required=True,
                        help="the name of the textual column in the csv file")
    parser.add_argument("--label_col_name", default=None, type=str, required=True,
                        help="the name of the label column in the csv file")
    parser.add_argument("--preprocess_data", default=None, type=str, required=True,
                        help="an indicator of preprocess tha data before training the model or not")
    parser.add_argument("--embeddings_file_path", default=None, type=str, required=True,
                        help=" the path of the word embeddings file in the Data folder")
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="the machine learning model to train")
    parser.add_argument("--results_file_name", default=None, type=str, required=True,
                        help="the name of the file which has the model's evaluation metrics in the trained_models folder")
    parser.add_argument("--saved_model_name", default=None, type=str, required=True,
                        help="the name of the saved trained model ")

    # reading the parameters that hte user send in the bash file
    args = parser.parse_args()
    training_ds_file_path = args.training_ds_file_path
    test_ds_file_path = args.test_ds_file_path
    text_col_name = args.text_col_name
    label_col_name = args.label_col_name
    preprocess_data = args.preprocess_data
    embeddings_file_path = args.embeddings_file_path
    model_name = args.model_name
    results_file_name = args.results_file_name
    saved_model_name = args.saved_model_name

    # Read datasets from the paths set by the user
    train_df = pd.read_csv("../../Data/Textual_data/"+training_ds_file_path, index_col=False)
    test_df = pd.read_csv("../../Data/Textual_data/"+test_ds_file_path, index_col=False)
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    #Model parameters
    batch_size = 32
    no_epochs = 100
    maxlen = 23
    embedding_size = 300 #word embedings size changes depending on the used word embeddings. More information about hte size of each emebeddings please refer to the documentation
    embedding_path = "../../Data/word_embeddings/"+embeddings_file_path # reading the path to the word embeddings set by the user
    embeddings_name = embeddings_file_path.split("/")[-1] #get the embeddings name from the path set by teh user
    print()
    if preprocess_data == "Yes":
        #Pre-process the data to be ready for training the model
        train_df[text_col_name] = train_df[text_col_name].apply(lambda x:helpers.noise_cleaning_preprocesing(x,remove_twitter_rev=False,remove_qoute=True,remove_stopwords=True,remove_punctuation=True))
        test_df[text_col_name] = test_df[text_col_name].apply(lambda x:helpers.noise_cleaning_preprocesing(x,remove_twitter_rev=False,remove_qoute=True,remove_stopwords=True,remove_punctuation=True))

    # Read and prepare the specified word emebddings to be used in the training process
    word_dictionary = helpers.get_word_dictionary(train_df, text_col_name)
    glove_cc_embeddings_matrix = embeddings.get_Glove_embeddings(
        embedding_path,
        word_dictionary,
        embedding_size)

    if model_name == "LSTM":
        # Train an LSTM model on teh dataset and word embeddings specified by teh user
        saver_path = "../../trained_models/LSTM/" # set the training model path
        results_file = "../Results/LSTM/" + results_file_name # set teh results file path
        models_training.LSTM_embeddings_model_training(train_df, test_df, text_col_name, label_col_name,
                                                                            maxlen
                                                                            , embedding_size, glove_cc_embeddings_matrix,
                                                                            embeddings_name,True
                                                                            , batch_size, no_epochs, saver_path,
                                                                            saved_model_name, results_file)
    elif model_name == "BILSTM":
        # Train a Bi-LSTM model on teh dataset and word embeddings specified by teh user
        saver_path = "../../trained_models/BiLSTM/"
        results_file = "../Results/BiLSTM/" + results_file_name
        models_training.BiLSTM_embeddings_model_training(train_df, test_df, text_col_name, label_col_name,
                                                                              maxlen
                                                                              , embedding_size, glove_cc_embeddings_matrix,
                                                                              embeddings_name,True
                                                                              , batch_size, no_epochs, saver_path,
                                                                             saved_model_name, results_file)
    elif model_name == "MLP":
        # Train an MLP model on teh dataset and word embeddings specified by teh user
        saver_path = "../../trained_models/MLP/"
        results_file = "../Results/MLP/" + results_file_name
        models_training.MLP_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                           maxlen
                                                                           , embedding_size, glove_cc_embeddings_matrix,
                                                                           embeddings_name,True
                                                                           , batch_size, no_epochs, saver_path,
                                                                           saved_model_name, results_file)
    elif model_name == "LR":
        # Train an LR model on teh dataset and word embeddings specified by teh user
        # LR model settings
        max_features = 10000  # Only consider the top 20k words
        saver_path = "../../trained_models/LR/"
        results_file = "../Results/LR/" + results_file_name
        models_training.LR_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                   max_features, batch_size, no_epochs,
                                                                   saver_path, saved_model_name, results_file)
if __name__ == "__main__":
    main()
