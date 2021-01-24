# Topic Modelling and Document Clustering in NLTK and Gensim


This project has two sections. The first section develops an algorithm to find similar documents in a document dataset, using `WordNet` and `path_similarity`. The second section develops a topic modeling algorithm. It uses Gensim's LDA (Latent Dirichlet Allocation) to identify 10 topics in a text dataset and the model is used to label new text documents. 


## Document Similarity

This section develops a set of functions that calculates the similarity between two documents by path_similarity and pos_tag. I use `convert_tag` function to convert nltk.pos_tag to the tag used by wordnet.synsets. Then, `doc_to_synsets` function extracts synsets, and `similarity_score` function calculates the maximum similarity between synsets by path_similarity. These two functions are used by `document_path_similarity` to find the path similarity between two documents.


```python
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


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

    0.554265873015873


<br>
`paraphrases` is a DataFrame which is provided with this repository. It contains the following columns: `Quality`, `D1`, and `D2`.
`Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).


```python
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quality</th>
      <th>D1</th>
      <th>D2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Ms Stewart, the chief executive, was not expec...</td>
      <td>Ms Stewart, 61, its chief executive officer an...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>After more than two years' detention under the...</td>
      <td>After more than two years in detention by the ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>"It still remains to be seen whether the reven...</td>
      <td>"It remains to be seen whether the revenue rec...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>And it's going to be a wild ride," said Allan ...</td>
      <td>Now the rest is just mechanical," said Allan H...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>The cards are issued by Mexico's consulates to...</td>
      <td>The card is issued by Mexico's consulates to i...</td>
    </tr>
  </tbody>
</table>
</div>





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

most_similar_docs()
```




    ('"Indeed, Iran should be put on notice that efforts to try to remake Iraq in their image will be aggressively put down," he said.',
     '"Iran should be on notice that attempts to remake Iraq in Iran\'s image will be aggressively put down," he said.\n',
     0.9753086419753086)



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

label_accuracy()
```




    0.8



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

lda_topics()
```




    [(0,
      '0.056*"edu" + 0.043*"com" + 0.033*"thanks" + 0.022*"mail" + 0.021*"know" + 0.020*"does" + 0.014*"info" + 0.012*"monitor" + 0.010*"looking" + 0.010*"don"'),
     (1,
      '0.024*"ground" + 0.018*"current" + 0.018*"just" + 0.013*"want" + 0.013*"use" + 0.011*"using" + 0.011*"used" + 0.010*"power" + 0.010*"speed" + 0.010*"output"'),
     (2,
      '0.061*"drive" + 0.042*"disk" + 0.033*"scsi" + 0.030*"drives" + 0.028*"hard" + 0.028*"controller" + 0.027*"card" + 0.020*"rom" + 0.018*"floppy" + 0.017*"bus"'),
     (3,
      '0.023*"time" + 0.015*"atheism" + 0.014*"list" + 0.013*"left" + 0.012*"alt" + 0.012*"faq" + 0.012*"probably" + 0.011*"know" + 0.011*"send" + 0.010*"months"'),
     (4,
      '0.025*"car" + 0.016*"just" + 0.014*"don" + 0.014*"bike" + 0.012*"good" + 0.011*"new" + 0.011*"think" + 0.010*"year" + 0.010*"cars" + 0.010*"time"'),
     (5,
      '0.030*"game" + 0.027*"team" + 0.023*"year" + 0.017*"games" + 0.016*"play" + 0.012*"season" + 0.012*"players" + 0.012*"win" + 0.011*"hockey" + 0.011*"good"'),
     (6,
      '0.017*"information" + 0.014*"help" + 0.014*"medical" + 0.012*"new" + 0.012*"use" + 0.012*"000" + 0.012*"research" + 0.011*"university" + 0.010*"number" + 0.010*"program"'),
     (7,
      '0.022*"don" + 0.021*"people" + 0.018*"think" + 0.017*"just" + 0.012*"say" + 0.011*"know" + 0.011*"does" + 0.011*"good" + 0.010*"god" + 0.009*"way"'),
     (8,
      '0.034*"use" + 0.023*"apple" + 0.020*"power" + 0.016*"time" + 0.015*"data" + 0.015*"software" + 0.012*"pin" + 0.012*"memory" + 0.012*"simms" + 0.011*"port"'),
     (9,
      '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')]



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

topic_distribution()
```




    [(0, 0.020003108),
     (1, 0.020003324),
     (2, 0.020001281),
     (3, 0.4967472),
     (4, 0.020004038),
     (5, 0.020004129),
     (6, 0.020002972),
     (7, 0.020002645),
     (8, 0.020003129),
     (9, 0.34322822)]




```python

```

This project is part of Applied Data Science with Python Specialization program at the University of Michigan. The program available from [here](https://www.coursera.org/learn/python-text-mining) and [here](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) . This code is only uploaded for educational purposes and it should not be used for submitting any homework or assignment.
