# Topic Modelling and Document Clustering in NLTK and Gensim


This project has two sections. The first section develops an algorithm to find similar documents in a document dataset, using `WordNet` and `path_similarity`. The second section develops a topic modeling algorithm. It uses Gensim's LDA (Latent Dirichlet Allocation) to identify 10 topics in a text dataset and the model is used to label new text documents. 


This project is part of Applied Data Science with Python Specialization program at the University of Michigan. The program available from [here](https://www.coursera.org/learn/python-text-mining) and [here](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) . This code is only uploaded for educational purposes and it should not be used for submitting any homework or assignment.


## Document Similarity

This section develops a set of functions that calculates the similarity between two documents by path_similarity and pos_tag. I use `convert_tag` function to convert nltk.pos_tag to the tag used by wordnet.synsets. Then, `doc_to_synsets` function extracts synsets, and `similarity_score` function calculates the maximum similarity between synsets by path_similarity. These two functions are used by `document_path_similarity` to find the path similarity between two documents.




```python
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    
    tokenized = nltk.word_tokenize(doc)
    pos_list = nltk.pos_tag(tokenized)
    list_of_synsets = []
    for tag in pos_list:
        wordnet_tag = convert_tag(tag[1])
        synset = wn.synsets(tag[0], wordnet_tag)
        if synset:
            list_of_synsets.append(synset[0])   
    return list_of_synsets


def similarity_score(s1, s2):
    
    synset_list = []
    for synset1 in s1:
        synset_1_list = []
        for synset2 in s2:
            t = synset1.path_similarity(synset2)
            if isinstance(t, float):
                synset_1_list.append(t)
        if synset_1_list:
            synset_list.append(max(synset_1_list))            
    return sum(synset_list)/len(synset_list)

def document_path_similarity(doc1, doc2):

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
```

### Testing similarity

Use this function calculates similarity score between doc1 and doc2.




```python
def test_document_path_similarity(doc1, doc2):
    return document_path_similarity(doc1, doc2)
doc1 = 'This is a function to test document_path_similarity.'
doc2 = 'Use this function to see if your code in doc_to_synsets \
        and similarity_score is correct!'
test_document_path_similarity(doc1, doc2)
```

<br>
`paraphrases` is a DataFrame which is provided with this repository. It contains the following columns: `Quality`, `D1`, and `D2`.

`Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).


```python
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()
```

### Find the most similar documents

Function `document_path_similarity` is coded to find which pair of documents has the maximum similarity score.




```python
def most_similar_docs():
    
    similarity_score_list = []
    for _, row in paraphrases.iterrows():
        similarity_score_list.append(document_path_similarity(row['D1'], row['D2']))
    similarity_score_list = np.array(similarity_score_list)
    t = np.argmax(similarity_score_list)
    return (paraphrases['D1'].iloc[t], paraphrases['D2'].iloc[t], similarity_score_list[t])
```

### Compare document_path_similarity function with labels

`label_accuracy` function comapres perfromance of our labaling function (i.e., `document_path_similarity`) with the labels provided in paraphrases. We consider the scores greater than 0.75 as paraphrase (1), and the rest as not paraphrase (0). Finally, accuracy of the classifier is reported using scikit-learn's accuracy_score.



```python
def label_accuracy():
    
    from sklearn.metrics import accuracy_score
    similarity_score_list = []
    for _, row in paraphrases.iterrows():
        similarity_score_list.append(document_path_similarity(row['D1'], row['D2']))
    similarity_score_list = [1 if i>0.75 else 0 for i in similarity_score_list]
    return accuracy_score(paraphrases['Quality'],similarity_score_list)
```

.

## Topic Modelling

This section uses Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`. First, I use gensim.models.ldamodel.LdaModel to estimate LDA model parameters on the corpus. Then, I create `ldamodel`model with `10 topics`,`passes=25`, and `random_state=34`.



```python
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
X = vect.fit_transform(newsgroup_data)
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics =10, id2word =id_map, passes=25, random_state=34)

```

Then, I Using `ldamodel` model to find a list of the 10 topics and their 10 most important words in each topic. This function uses `print_topics` to extract the topics.


```python
def lda_topics():
    
    return ldamodel.print_topics(num_topics =10, num_words =10)
```

### Testing a new document

Now, we can use `ldamodel` model to cluster some new documents (e.g., `new_doc`). `topic_distribution` function reads the new document, vectorizes it, and converts the sparse matrix to gensim corpus. It returns a list which shows the probability of belonging to each topic. 



```python
new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]
```


```python
def topic_distribution():
    
    test_x = vect.transform(new_doc)
    test_corpus = gensim.matutils.Sparse2Corpus(test_x, documents_columns=False)
    return list(ldamodel[test_corpus])[0]

```


