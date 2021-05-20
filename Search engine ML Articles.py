import pandas as pd
import requests
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from collections import defaultdict, OrderedDict
import numpy as np
import networkx as nx

#Class database to retrive offline documents and build inverted index
class database:
    def invertedIndex(self):
        len=0
        i = index_nlp()
        invertedIndex = defaultdict(list)
        with open('dataset.csv', newline='', encoding="utf8") as f:
            documents = csv.DictReader(f)
            for document in documents:                                  #each row is a document
                len += 1
                docs_inverted = i.buildIndex(document)
                for key, value in docs_inverted.items():
                    invertedIndex[key].append(value)                    #build inverted index
        return invertedIndex, len


#Class crawler to retrive documents from web pages
class crawler:

    def __init__(self):
        self.to_visit = list()
        self.visited = set()

    def fetch(self, url):
        print('now fetching.. ', url)
        res = requests.get(url)

        return res.content

    def get_current_url(self):
        res = self.to_visit.pop(0)

        while res in self.visited:
            print('already visited', res)
            res = self.to_visit.pop(0)

        return res

    def get_links(self, current_url, content):
        urls = re.findall('<a href="([^"]+)">', str(content))
        print('urls are', urls)
        for url in urls:
            if url[0] == '/':
                url = current_url + url[1:]
            pattern = re.compile('https?')
            if pattern.match(url):
                self.to_visit.append(url)

    def crawl(self, url, depth=100):
        self.to_visit.append(url)
        while len(self.visited) <= depth:
            for id in range(0, len(self.visited)):
                current_url = sel1f.get_current_url()                                                # get url from web page
                content = self.fetch(current_url)                                                   # retrive content from web page
                p_content = re.findall('<p>(.*?)</p>', str(content))                                # find paragraph of page
                data = pd.DataFrame({'Doc_ID': id, 'URL': current_url, 'Content': p_content})       # make it as dataFrame
                dataset = data.to_csv('dataset.csv', index=False)                                   # save final dataset file
                self.visited.add(current_url)
                self.get_links(current_url, content)
        return dataset


class page_rank:

    def __init__(self, num):
        self.totalNodes = num;
        self.graph = np.zeros((self.totalNodes, self.totalNodes))
        self.pagerank = np.zeros(self.totalNodes)

    def print_ranks(self):
        for k in range(self.totalNodes):
            print(" Page Rank of ", k, " is :\t", self.pagerank[k])

    def calc_out_going_links(self, nodeNumber):

        outgoingLinks = 0
        # outgoingLinks = np.sum(self.path[nodeNumber][:])

        for k in range(self.totalNodes):
            if (self.graph[nodeNumber][k] == 1):
                outgoingLinks += 1

        return outgoingLinks

    def calc(self):

        outgoingLinks = 0
        dampingFactor = 0.85
        tempPageRank = np.zeros(self.totalNodes)

        initialPageRank = 1.0 / self.totalNodes;
        print(" Total Number of Nodes :", self.totalNodes, "\t Initial PageRank  of All Nodes :", initialPageRank, "\n")

        # 1st ITERATION  _ OR _ INITIALIZATION PHASE
        for k in range(self.totalNodes):
            self.pagerank[k] = initialPageRank

        print("\n Initial PageRank Values , Step 1 \n")

        self.print_ranks()

        iterations_number = 10

        for i in range(iterations_number):  ##Iterations

            # Store the PageRank for All Nodes in Temporary Array
            for k in range(self.totalNodes):
                tempPageRank[k] = self.pagerank[k]
                self.pagerank[k] = 0

            for internalNodeNumber in range(self.totalNodes):
                for externalNodeNumber in range(self.totalNodes):
                    if (self.graph[externalNodeNumber][internalNodeNumber] == 1):
                        outgoingLinks = self.calc_out_going_links(externalNodeNumber);
                        # Calculate PageRank
                        self.pagerank[internalNodeNumber] += tempPageRank[externalNodeNumber] * (1.0 / outgoingLinks);

                print("\n After Step ", i, " \n");

                self.print_ranks()

        # Add the Damping Factor to PageRank
        for k in range(self.totalNodes):
            self.pagerank[k] = (1 - dampingFactor) + dampingFactor * self.pagerank[k]

        # Display PageRank
        self.print_ranks()



#class index nlp for processing the documents
class index_nlp:
    def buildIndex(self, document):
        res = dict()
        terms = word_tokenize(document['Content'])
        for j in range(len(terms)):
            term = terms[j].lower()
            term = stemmer.stem(term)
            if term not in stop_words:
                if term in document['Content'].lower():
                    if(term in res.keys()):
                        res[term] = [res.get(term)[0], res.get(term)[1]+1, res.get(term)[2]+", "+str(j), res.get(term)[3],res.get(term)[4]]
                    else:
                        res[term] = [document['Doc_ID'], 1, str(j+1), document['URL'], len(terms)]
                    #each res(i) contain [id(i),frequancy(i),postions(i),url(i),length of doc(i)]
        return res                                                          # return each document after processing

class TF_IDF:
    def compute_tf(self, term, invertedIndex):
        l=[]
        for j in range(0, len(invertedIndex.get(term))):
            tf= invertedIndex.get(term)[j][1]/invertedIndex.get(term)[j][4]
            l.append(tf)
        return(l)

    def compute_idf(self, term, invertedIndex,length_of_docs):
        idf = np.log(length_of_docs/len(invertedIndex.get(term))+1)
        return idf

    def compute_tf_idf(self,word,invertedIndex,length_of_docs):
        l=[]
        tf= self.compute_tf(word,invertedIndex)
        idf = self.compute_idf(word,invertedIndex,length_of_docs)
        for i in tf:
                tf_idf = i*idf
                l.append(tf_idf)
        for j in range(0, len(invertedIndex.get(word))):
            invertedIndex.get(word)[j].append(l[j])
        return invertedIndex


class serach:
  def searchSynset(self,word):
      for w in wn.synsets(word):
          print('Term {0} has word synonym :{1}'.format(word, w.name()))
          print("----------------------------------------------------")
  def searchDict(self, word, invertedIndex,length_of_docs):
      s=editDistance()
      tf_idf= TF_IDF()
      tokens = word_tokenize(word)
      terms = self.booleanSearchQueries(tokens)
      for i in range(len(terms)):
          term = terms[i].lower()
          term = stemmer.stem(term)
          if term not in stop_words:
           if term in invertedIndex:
               invertedIndex = tf_idf.compute_tf_idf(term, invertedIndex, length_of_docs)
               invertedIndex.get(term).sort(key=lambda x: x[5], reverse=True)
               for j in range(0, len(invertedIndex.get(term))):
                    print('Term {0} is found on:\nDocument ID: {1}\nWith frequency of: {2}\nIn position: {3}\nDocument URL: {4}\nTF_IDF: {5}'
                          .format(term, invertedIndex.get(term)[j][0], invertedIndex.get(term)[j][1],
                            invertedIndex.get(term)[j][2],invertedIndex.get(term)[j][3],invertedIndex.get(term)[j][5]))
                    print("----------------------------------------------------")
           else:
                   similarity = s.med(word, invertedIndex)
                   print("Did you mean: {0}?\nEnter y for YES or n for NO".format(similarity))
                   cho = input()
                   if cho == 'y':
                       self.searchDict(similarity, invertedIndex, length_of_docs)
                   else:
                       return


  def searchOntology(self,term):
      dictOntology={
          'Artificial intelligence':'The intelligence of machines and the branch of computer science that aims to create it.',
          'Rational Agent' :'Within artificial intelligence,is one that maximizes its expected utility, given its current knowledge.',
          'Turing Test':'A field of computer science and linguistics concerned with the interactions between computers and human languages.',
          'Natural Language Processing':'A field of computer science and linguistics concerned with the interactions between computers and human languages.',
          'Knowledge Representation (KR)':'Translation of information into symbols to facilitate inferencing from those information elements, and the creation of new elements of information.',
          'Machine Learning':'A scientific discipline concerned with the design and development of algorithms that allow computers to evolve behaviors based on empirical data, such as from sensor data or databases.',
          'Computer Vision':'A field that includes methods for acquiring, processing, analysing, and understanding images and, in general, high-dimensional data from the real world in order to produce numerical or symbolic information.',
          'Robotics':'The branch of technology that deals with the design, construction, operation, structural disposition, manufacture and application of autonomous machines and computer systems for their control, sensory feedback, and information processing',
          'Probability':'Besides logic and computation, the third great contribution of mathematics to AI is the theory of _________. The Italian Gerolamo Cardano (1501-1576) first framed the idea, describing it in terms of the possible outcomes of gambling events.',
          'Game':'A scenario in which the actions of one player can significantly affect the utility of another (either positively or negatively).',
          'Neuron':'An electrically excitable cell that processes and transmits information by electrical and chemical signaling',
          'Perceptron':'An algorithm for supervised classification of an input into one of two possible outputs. It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector describing a given input.',
          'Expert Systems':'A computer system that emulates the decision-making ability of a human expert and are designed to solve complex problems by reasoning about knowledge, like specialist, and not by following the procedure of a developer as is the case in conventional programming.',
          'Back-Propagation':'A common method of training artificial neural networks so as to minimize the objective function.',
          'Data Mining':'A process that results in the discovery of new patterns in large data sets. Utilizes methods at the intersection of artificial intelligence, machine learning, statistics, and database systems.',
          'Bayesian Network':'A probabilistic graphical model (a type of statistical model) that represents a set of random variables and their conditional dependencies via a directed acyclic graph (DAG).',
          'Optimization':'The selection of a best element from some set of available alternatives.',
          'AI winter':'A period of reduced funding and interest in artificial intelligence research. The field has experienced several cycles of hype, followed by disappointment and criticism, followed by funding cuts, followed by renewed interest years or decades later. There were two major instances in 1974-80 and 1987-93.',
          'Training Set':'Example input-output pairs, from which to discover a hypothesis',
          'Test Set':'Examples distinct from training set, used to estimate accuracy',
          'Cross-validation':'Randomly split the data into a training set and a test set',
          'Overfitting':'Choose an over-complex model based on irrelevant data patterns.',
          'Big Data':'Big Data is any data that is expensive to manage and hard to extract value from.',
          'Data Science':'The analysis of data using the scientific method',
          'Data Scientist':'if the scientific method is applied ( a statistically controlled six-step process: question, research, hypothesize, experiment, analyze, conclude), the professionals doing the work qualify as scientists.',
          'Hadoop':'Apache Hadoop is an open-source software framework for storage and large-scale processing of data-sets on clusters of commodity hardware. It is commonly used in "Hadoop clusters," which are purpose-designed computational clusters.',
          'Data mining':'the process of finding certain patterns or information from data sets',
          'data analyst':'someone analysing, modelling, cleaning or processing data',
          'model':'Defines the relationship between features and label.',
          'training':'Showing the model labeled examples and enable the model to gradually learn the relationships between features and label.',
          'regression model':'A type of model that outputs continuous (typically, floating-point) values. Compare with classification models, which output discrete values, such as "day lily" or "tiger lily."',
          'classification model':'A type of machine learning model for distinguishing among two or more discrete classes. For example, a natural language processing classification model could determine whether an input sentence was in French, Spanish, or Italian.',
          'tensor':'N-dimensional (where N could be very large) data structures, most commonly scalars, vectors, or matrices. The elements of it can hold integer, floating-point, or string values.',
          'weight':'A coefficient for a feature in a linear model, or an edge in a deep network. The goal of training a linear model is to determine the ideal _______ for each feature. If a ______ is 0, then its corresponding feature does not contribute to the model.',
          'mean squared error(MSE)':'The average squared loss per example. _____ is calculated by dividing the squared loss by the number of examples. The values that TensorFlow Playground displays for "Training loss" and "Test loss"',
      }
      dictOntology = {k.lower(): v for k, v in dictOntology.items()}
      term=term.lower()
      s=editDistance()
      if term not in stop_words:
          if term in dictOntology:
                  print('{0}: {1}\n'.format(term, dictOntology.get(term)))
                  print("----------------------------------------------------")
          else:
              similarity = s.med(term, dictOntology)
              print("Did you mean: {0}?\nEnter y for YES or n for NO".format(similarity))
              cho = input()
              if cho == 'y':
                  self.searchOntology(similarity)
              else:
                return 0
  def booleanSearchQueries(self,terms):
      if 'or | OR' in terms:
          terms.remove('or | OR')
          return terms
      elif 'and | AND' in terms:
          terms.remove('and | AND')
          return terms
      else:
          return terms


class editDistance:
    def med(self, word, invertedIndex):
        l={}
        for key in invertedIndex.keys():
            res = np.zeros((len(word), len(key)))
            for i in range(len(word)):
                res[i, 0] = i

            for j in range(len(key)):
                res[0, j] = j
            for i in range(1, len(word)):
                for j in range(1, len(key)):
                    c1 = res[i, j - 1] + 1
                    c2 = res[i - 1, j] + 1

                    if (word[i - 1] == key[j - 1]):
                        c3 = res[i - 1, j - 1]
                    else:
                        c3 = res[i - 1, j - 1] + 2

                    res[i, j] = min(c1, c2, c3)
            l[key] = res[len(word) - 1][len(key) - 1]
            l = OrderedDict(sorted(l.items(), key=lambda x: x[1]))
        return list(l.keys())[0]
#------------------------------------Intilizing-----------------------------------
#c = crawler()
#dict_content = c.crawl('https://medium.com/tag/artificial-intelligence')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
db=database()
invertedIndex, length = db.invertedIndex()
s = serach()
G = nx.barabasi_albert_graph(60,41)
pr = nx.pagerank(G,0.4)
nodes = 5
p = page_rank(nodes)
graph = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0]])

p.graph = graph
#------------------------------------User interface-----------------------------------
print("\nWelcome to Haythom. What would you like to search?")
choice = ''
while choice != 'q':
    print("\n[1] Enter 1 to search in documents.")
    print("[2] Enter 2 to search for word synonyms.")
    print("[3] Enter 3 to search in our ontology.")
    print("[q] Enter q to quit.")


    choice = input("\nWhat would you like to do? ")

    if choice == '1':
        s.searchDict(input("Enter term to serach :"), invertedIndex,length)
    if choice == '2':
        s.searchSynset(input("Enter term to serach :"))
    if choice == '3':
        s.searchOntology(input("Enter term to serach :"))
    elif choice == 'q':
        print("\nThanks for searching on Haythom.\n")

#------------------------------------Testing URLs-----------------------------------
# c.crawl('https://www.businessinsider.com/artificial-intelligence')
# c.crawl('https://openai.com/')
# c.crawl('http://artent.net/')
# c.crawl('http://aiweekly.co/')