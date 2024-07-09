# Medical-Information-Retrieval-System
# Project Description

A medical information retrieval system is specialized software that consists of methodologies and technologies that allow for both efficient and accurate retrieval of relevant medical information from diverse sources that potentially include but are not limited to the web, articles, journals, social media, and hospital records. Information related to health is one of the most searched topics on the internet and is of vast importance to an array of users, such as medical practitioners, researchers, patients, and their families.

The main objective is to implement a medical information retrieval system, i.e., a search engine that is specific to the medical domain, that will retrieve top-ranked relevant documents accurately from the dataset on the basis of the user’s input query to ensure that the user is able to obtain relevant results to cater to varying medical information needs, whether related to diseases themselves, their treatment and medications, or research findings.

# Methodology

## Dataset
The dataset being used is “A Full-Text Learning to Rank Dataset for Medical Information Retrieval” [1].

## Preprocessing
1. **Tokenization**  
   Tokenization will be performed to split both the query and the words in the document.
   
2. **Stop word removal**  
   Stop words will be removed in accordance with a predefined list of stop words.
   
3. **Lemmatization**  
   Words will be reduced to their lemma.
   
4. **Stemming**  
   Stemming will be performed based on the Porter Stemmer.

## Indexing
A hash table will be used to index both documents and search queries.

## Algorithmic Model
We will use a method called the TF-IDF weight method, which is a measure of how important each word is in a document compared to a whole collection of documents. TF-IDF stands for Term Frequency-Inverse Document Frequency. Furthermore, normalization will be performed, and then the cosine similarity will be calculated to determine the relevance.

## Retrieval
A heap implemented using a priority queue will be used to retrieve top-ranked results.

## Relevance Feedback
Recall is an important factor in medical settings, which is why the Rocchio algorithm will be used to implement relevance feedback in which users can provide feedback on search results, classifying them into relevant and non-relevant.

## Reference
[1] V. Boteva, D. Gholipour, A. Sokolov, and S. Riezler, “A Full-Text Learning to Rank Dataset for Medical Information Retrieval,” Proceedings of the 38th European Conference on Information Retrieval, 2016. [Online serial]. Available: [https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf](https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf). [Accessed Mar. 4, 2024].

