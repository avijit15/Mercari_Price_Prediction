from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import mean_squared_error
import nltk
nltk.download('wordnet')
from scipy.sparse import hstack

global brand_encode_dict
global cat_1_vectorizer
global categoty_2_encode_dict
global categoty_3_encode_dict
global vectorizer_name
global vectorizer_item
global final_model

with open('brand_encode.pickle', 'rb') as f:
    brand_encode_dict = pickle.load(f)

with open('cat_1_vector.pickle', 'rb') as f:
    cat_1_vectorizer = pickle.load(f)

with open('category_2_encode.pickle', 'rb') as f:
    categoty_2_encode_dict = pickle.load(f)

with open('category_3_encode.pickle', 'rb') as f:
    categoty_3_encode_dict = pickle.load(f)

with open('vectorizer_name_tfidf.pickle', 'rb') as f:
    vectorizer_name = pickle.load(f)

with open('vectorizer_item_tfidf.pickle', 'rb') as f:
    vectorizer_item = pickle.load(f)

with open('best_lgb_final.pickle', 'rb') as f:
    final_model = pickle.load(f)


def basic_process_for_cat_and_brand(text):
    ''' Basic Preprocessing for Category and brand_name'''

    string = str(text)
    string = string.replace('&', '')
    string = string.replace('-', ' ')
    string = string.replace('\'s', '')

    # Removing non alphanumeric charecters
    string = re.sub('[^A-Za-z0-9é]+', ' ', string)
    string = '_'.join(string.split())

    return string


def decontract_text(text):
    string = str(text)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'m", " am", string)
    return string


def remove_stopwords(text):
    StopWords = stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                             "you've", \
                             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                             'himself', \
                             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
                             'them', 'their', \
                             'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                             'these', 'those', \
                             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                             'having', 'do', 'does', \
                             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                             'until', 'while', 'of', \
                             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                             'during', 'before', 'after', \
                             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                             'under', 'again', 'further', \
                             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                             'both', 'each', 'few', 'more', \
                             'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
                             'very', \
                             's', 't', 'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll',
                             'm', 'o', 're', \
                             've', 'y']
    string = str(text).lower()
    string = string.replace('\\r', ' ')
    string = string.replace('\\n', ' ')
    string = string.replace('\\t', ' ')
    string = string.replace('\\"', ' ')
    string = re.sub('[^A-Za-z0-9]+', ' ', string)
    string = decontract_text(string)
    sentence = []
    for word in string.split():
        if word not in StopWords:
            sentence.append(word)
    return ' '.join(sentence)


def final_fun_1(test_data):
    '''The input to the function will be a lsit of list where, each list indicates one row\
       The input structure should be like below\
       [[name, item_condition_id, brand_name,shipping, item_description, category_1, category_2, category_3]]'''

    ################################################################################################

    ## Function for Lemmatization
    lam = WordNetLemmatizer()

    def lamitizing(text):
        tokens = text.split()
        lam_list = []
        for token in tokens:
            lam_list.append(lam.lemmatize(token))
        return ' '.join(lam_list)

    ################################################################################################

    ## Function for brand_name encode

    def brand_name_encode(x):
        # for values which are present in train data, for that it will return its corresponding value from dict
        if x in brand_encode_dict.keys():
            return brand_encode_dict[x]
        # the values which are not present in train data, for that it will return 0
        else:
            return 0

    ##################################################################################################

    ## Function for cat_2 encoding

    def category_2_encode(x):
        # for values which are present in train data, for that it will return its corresponding value from dict
        if x in categoty_2_encode_dict.keys():
            return categoty_2_encode_dict[x]
        # the values which are not present in train data, for that it will return 0
        else:
            return 0

    #################################################################################################

    ## Function for cat_3 encoding

    def category_3_encode(x):
        # for values which are present in train data, for that it will return its corresponding value from dict
        if x in categoty_3_encode_dict.keys():
            return categoty_3_encode_dict[x]
        # the values which are not present in train data, for that it will return 0
        else:
            return 0
    print(0)
    ##################################################################################################

    if type(test_data) != list or type(test_data[0]) != list:
        print('Please give input as a list of list where each inner list indicates a row')
    else:
        data = pd.DataFrame({'name': [i[0] for i in test_data], 'item_condition_id': [i[1] for i in test_data], \
                             'category_name': [i[2] for i in test_data],
                             'brand_name': [i[3] for i in test_data], \
                             'shipping': [i[4] for i in test_data],
                             'item_description': [i[5] for i in test_data]})

        ##Preprocess Category
        data['category_1'] = data.category_name.apply(lambda x: str(x).split('/')[0])
        data['category_2'] = data.category_name.apply(lambda x: str(x).split('/')[1])
        data['category_3'] = data.category_name.apply(lambda x: str(x).split('/')[2])

        data.drop('category_name', axis=1, inplace=True)

        data['category_1'] = data.category_1.apply(basic_process_for_cat_and_brand)
        data['category_2'] = data.category_2.apply(basic_process_for_cat_and_brand)
        data['category_3'] = data.category_3.apply(basic_process_for_cat_and_brand)

        data['category_2_encode'] = data['category_2'].apply(category_2_encode)
        data['category_3_encode'] = data['category_3'].apply(category_3_encode)

        cat1_ohe = cat_1_vectorizer.transform(data['category_1']).toarray()

        ##Preprocess Brand Name

        data['brand_name'] = data.brand_name.apply(basic_process_for_cat_and_brand)
        data['brand_name_encode'] = data['brand_name'].apply(brand_name_encode)
        print(1)
        ###################################################################################################

        ##Preprocess Name
        data['name'] = data.name.apply(remove_stopwords)
        data['name'] = data.name.apply(lamitizing)

        data_name_tfidf = vectorizer_name.transform(data['name']).tocsr()

        #################################################################################################

        ##Preprocess item_description
        data['item_description'] = data.item_description.apply(remove_stopwords)
        data['item_description'] = data.item_description.apply(lamitizing)

        data_item_tfidf = vectorizer_item.transform(data['item_description']).tocsr()

        ## Tokenization and padding for item_description
        print(2)
        ##Adding all the non-text data
        final_data = hstack(
            (data['item_condition_id'].values.reshape(-1, 1), data['shipping'].values.reshape(-1, 1), \
             data['brand_name_encode'].values.reshape(-1, 1), cat1_ohe,
             data['category_2_encode'].values.reshape(-1, 1), \
             data['category_3_encode'].values.reshape(-1, 1), \
             data_name_tfidf, data_item_tfidf)).tocsr()
        print(3)
        ###################################################################################################

        # final_model = models.load_model('Lstm_Dense.h5',custom_objects={'rmse':rmse} )
        y_predict = final_model.predict(final_data)
        print(4)
        y_predict = [np.exp(i) - 1 for i in y_predict]

        dic = dict(zip([i[0] for i in test_data], y_predict))

        return dic

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])  #To render home page
def home_page():
    return render_template('index.html',result = {})

@app.route('/predict',methods=['POST']) #To take a route
def predict():
    if request.method == 'POST' :

        name = request.form['name']
        category_name = request.form['category_name']
        brand_name = request.form['brand_name']
        item_description = request.form['item_description']
        item_condition_id = int(request.form['item_condition_id'])
        shipping = int(request.form['shipping'])
        test_data = [[name,item_condition_id,category_name,brand_name,shipping,item_description]]
        dic = final_fun_1(test_data)

    return render_template('index.html', result=dic)




if __name__ == '__main__':
    app.run(debug = False)
