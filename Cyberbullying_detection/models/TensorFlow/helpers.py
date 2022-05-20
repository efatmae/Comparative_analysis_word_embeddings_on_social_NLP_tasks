from keras.preprocessing.text import Tokenizer
import contractions_dict
from collections import Counter
from string import punctuation
from nltk.tokenize import word_tokenize

'''This python file contain functions that help in the training process of Tensorflow platform models'''
def _data_generator(x, y, num_features, batch_size):
    """Generates batches of vectorized texts for training/validation.

    # Arguments
        x: np.matrix, feature matrix.
        y: np.ndarray, labels.
        num_features: int, number of features.
        batch_size: int, number of samples per batch.

    # Returns
        Yields feature and label data in batches.
    """
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield x_batch, y_batch

def get_word_dictionary(dataset, label):
    '''
    This function generates a word dictionary from the chosen dataset
    Args:
        dataset: pandas dataframe of the training dataset
        label: name of the label columns - string

    Returns: the word dictionary of the dataset

    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataset[label])
    return tokenizer.word_index

def noise_cleaning_preprocesing(text, remove_twitter_rev, remove_qoute, remove_stopwords, remove_punctuation):
    '''
        This function preprocess the textual dataset before training a ML model on
        Args:
            text: the text to be assessed against the ML model - string
            remove_twitter_rev: indicator to remove twitter twitter handles from the text - boolean
            remove_qoute: indicator to remove single and double qoutes  from the text - boolean
            remove_stopwords: indicator to remove stop words from the text - boolean
            remove_punctuation: indicator to remove the punctuation from the text - boolean

        Returns:
            text: the preprocessed text - string
        '''
    import preprocessor as p
    from contractions_dict import CONTRACTION_MAP
    import unicodedata
    from nltk.corpus import stopwords
    import re

    if remove_twitter_rev == True:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
    else:
        p.set_options(p.OPT.URL, p.OPT.MENTION)

    def camel_case_split(identifier):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def remove_tags(text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def custome_remove_punctuation(words):
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        return stripped

    def remove_unwanted_signs(text):
        if remove_qoute == True:
            text = text.replace('"', "")
        # text = text.replace('\\', "")
        # text = text.replace('!', "")
        # text = text.replace('/', "")
        # text = text.replace('*', "")
        text = text.replace("\n", "")
        # text = text.replace(":","")
        text = text.replace("#", "")
        text = text.replace("&amp", "and")
        text = text.replace("&lt", "<")
        text = text.replace("&gt", ">")
        return text

    def custome_remove_stop_words(words):
        keep_words = ["you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they", "them", "their",
                      "theirs"]
        stop_words = set(stopwords.words('english'))
        stop_words = [word for word in stop_words if not word in keep_words]
        words = [w for w in words if not w in stop_words]
        return words

    clean_text = remove_unwanted_signs(text)
    clean_text = " ".join(camel_case_split(clean_text))
    encoded_string = clean_text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    clean_text = remove_tags(decode_string)
    clean_text = remove_accented_chars(clean_text)
    clean_text = p.clean(clean_text)
    clean_text = expand_contractions(clean_text)

    tokens = word_tokenize(clean_text)
    words = [word.lower() for word in tokens]
    words = [word for word in words if not word.isdigit()]  # remove numbers

    if remove_stopwords == True:
        words = custome_remove_stop_words(words)

    if remove_punctuation == True:
        words = custome_remove_punctuation(words)

    text = " ".join(words)
    text = re.sub(' +', ' ', text)
    return text