import contractions
from flask import Flask
from flask import request
from flask import render_template

import pandas as pd
import json
from annoy import AnnoyIndex
import string,re
from flask import jsonify
from sentence_transformers import models
from sentence_transformers import SentenceTransformer 

import spacy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__, static_folder="static")


def data_preprocess(text):
    #removing contractions
    text=contractions.fix(text)
    #removing url and html links
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = re.compile('<.*?>')
    text=html_pattern.sub(r'', text)
    text=url_pattern.sub(r'', text)
    #replacing underscore from text
    text=text.replace('_',' ')
    #To remove the punctuations
    text = text.translate(str.maketrans('','',string.punctuation))
    #will consider only alphabets and numerics
    pat = r'[^a-zA-z0-9]' 
    text=re.sub(pat, ' ', text)  
    #will replace newline with space
    text = re.sub("\n"," ",text)
    #will convert to lower case and will split and join the words
    text=' '.join(text.split())
    text=text.lower()
    return text

def data_prep():

    metadata = pd.read_csv('data/csv_for_search.csv',  low_memory=False)
    return metadata

cord_data=data_prep()


## Step 1: use an existing language model
word_embedding_model = models.Transformer('lordtt13/COVID-SciBERT')

## Step 2: use a pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

## Join steps 1 and 2 using the modules argument
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Annoy Indexing
f = 768

u2 = AnnoyIndex(f, 'angular')
u2.load(r'data\ann_index.ann') # super fast, will just mmap the file




@app.route("/", methods=['GET'])
def land():
    return render_template('index.html')


@app.route("/similar", methods=['GET', 'POST'])
def similar():

    data_req = json.loads(request.data)
    data=data_req['query']
    print(data)

    data=data_preprocess(data)

    data = model.encode(data)

    # Roberta Pretrained model
    my_result_out2=[]
    x,dist=u2.get_nns_by_vector(data, 5, search_k=-1, include_distances=True)
    print(x,dist)
    for i,j in zip(x,dist):
        temp_list=list(cord_data.loc[i,['title','abstract','url']].values)
        temp_list.append(1-((j**2) / 2))
        my_result_out2.append(temp_list)

    df_50=pd.DataFrame(my_result_out2,columns=['title','abstract','url','Similarity'])
    df1=df_50.head(3)

    table1 = df1.to_html()

    return jsonify(value1= table1)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=8000,debug=True)