# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER

# COMMAND ----------

# MAGIC %md # HW 5 - Page Rank
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC In Weeks 8 and 9 you discussed key concepts related to graph based algorithms and implemented SSSP.   
# MAGIC In this final homework assignment you'll implement distributed PageRank using some data from Wikipedia.
# MAGIC By the end of this homework you should be able to:  
# MAGIC * ... __compare/contrast__ adjacency matrices and lists as representations of graphs for parallel computation.
# MAGIC * ... __explain__ the goal of the PageRank algorithm using the concept of an infinite Random Walk.
# MAGIC * ... __define__ a Markov chain including the conditions underwhich it will converge.
# MAGIC * ... __identify__ what modifications must be made to the web graph in order to leverage Markov Chains.
# MAGIC * ... __implement__ distributed PageRank in Spark.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__ 

# COMMAND ----------

# MAGIC %md # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.   

# COMMAND ----------

# init script to create the blob URL
blob_container = "team07"
storage_account = "team07"
secret_scope = "team07"
secret_key = "team07"
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

# generates the SAS token
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# RUN THIS CELL AS IS. 
tot = 0
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
for item in dbutils.fs.ls(DATA_PATH):
  tot = tot+item.size
tot
# ~4.7GB

# COMMAND ----------

# RUN THIS CELL AS IS. You should see all-pages-indexed-in.txt, all-pages-indexed-out.txt and indices.txt in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

sc = spark.sparkContext
spark

# COMMAND ----------

# MAGIC %md # Question 1: Distributed Graph Processing
# MAGIC Chapter 5 from Lin & Dyer gave you a high level introduction to graph algorithms and concerns that come up when trying to perform distributed computations over them. The questions below are designed to make sure you captured the key points from this reading and your async lectures. 
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Give an example of a dataset that would be appropriate to represent as a graph. What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? What would the average "in-degree" of a node mean in the context of your example? 
# MAGIC 
# MAGIC * __b) short response:__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm? *(__HINT__: Do not respond in terms of any specific algorithm. Think in terms of the nature of the graph datastructure itself).*
# MAGIC 
# MAGIC * __c) short response:__ Briefly describe Dijskra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?
# MAGIC 
# MAGIC * __d) short response:__ How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?

# COMMAND ----------

# MAGIC %md ### Q1 Student Answers:
# MAGIC > __a)__ A co-occurance matrix which is used to denote the similarity between a pair of words can be represented by a graph. In this data structure, the nodes represent words and the edges represent connections between words that co-occur on a given context. This is an undirected graph. The average "in-degree" of a node represents the number of relationships each word on our vocabulary has on average with the others.
# MAGIC 
# MAGIC > __b)__ Graph algorithms are iterative, requiring many map-reduce iteractions to converge. At each iteraction, in addition to nodes' computations, the graph structure itself needs to be passed from the mappers to the reducers, which makes each iteraction potentially expensive (lots of KV pairs being emmitted and shuffled through the network). Moreover in a parallelized implementation there is not the possibility of using global variables to keep state across nodes.
# MAGIC 
# MAGIC > __c)__ Dijkstra's algorithm solves the single-source shortest-path problem for any weighted, directed graph with non-negative weights. The algorithm operates by iteratively selecting the node with the lowest current distance from the priority queue. At each iteration, the algorithm "expands" that node by traversing the adjacency list of the selected node to see if any of those nodes can be reached with a path of a shorter distance. The algorithm terminates when the priority queue is empty, or equivalently, when all nodes have been considered. The key to Dijkstra's algorithm is the priority queue that maintains a globally sorted list of nodes by current distance. This is not possible in a MapReduce framework, as the programming model does not provide a mechanism for exchanging global state across nodes
# MAGIC 
# MAGIC > __d)__ Instead of using a priority queue, the parallel breadth-first search adopts a brute force approach where all potential paths to a given node will be explored but only the shortest distance will be kept. At each iteration, the algorithm attempts to recompute distances to all nodes, but in reality only useful work is done along the search frontier. Inside the search frontier, the algorithm is simply repeating previous computations. Outside the search frontier, the algorithm hasn't discovered any paths to nodes so no meaningful work is done. This results in a lot of inneficiency compared with Dikstra's sequential processing.

# COMMAND ----------

# MAGIC %md # Question 2: Representing Graphs 
# MAGIC 
# MAGIC In class you saw examples of adjacency matrix and adjacency list representations of graphs. These data structures were probably familiar from HW3, though we hadn't before talked about them in the context of graphs. In this question we'll discuss some of the tradeoffs associated with these representations. __`NOTE:`__ We'll use the graph from Figure 5.1 in Lin & Dyer as a toy example. For convenience in the code below we'll label the nodes `A`, `B`, `C`, `D`, and `E` instead of $n_1$, $n_2$, etc but otherwise you should be able to follow along & check our answers against those in the text.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/Lin-Dyer-graph-Q1.png?raw=true" width=50%>
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph.
# MAGIC 
# MAGIC * __b) short response:__ Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code to complete the function `get_adj_matr()`.
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code to complete the function `get_adj_list()`.

# COMMAND ----------

# MAGIC %md ### Q2 Student Answers:
# MAGIC > __a)__ The graph described in the above figure is sparse. The problem with adjacency matrix is 1) it always has O(n^2) space requirement irrespective of the graph being sparse or dense and 2) most of the cells are zero in case of a sparse graph. Graphs are also represented as adjacency lists that associates each node in the graph with the collection of its neighbouring nodes or links. For eaxmple, since n1 is connected to n2 and n4, they occur in the adjacency list of n1. The problem with adjacency lists is that operations on incoming links can be very expensive, while in an adjacency matrix this would be easily solvable by a column scan of the matrix.
# MAGIC 
# MAGIC > __b)__ This graph is directed. For a directed graph, the adjacency matrix represents only neighbors that can be reached via outgoing edges. It can be nonsymmetric. If a graph is undirected the adjacency matrix is symmetric with zeros on the diagonal.

# COMMAND ----------

# part a - a graph is just a list of nodes and edges (RUN THIS CELL AS IS)
TOY_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
             'edges':[('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'D'), 
                      ('D', 'E'), ('E', 'A'),('E', 'B'), ('E', 'C')]}

# COMMAND ----------

# part a - simple visualization of our toy graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY_GRAPH['nodes'])
G.add_edges_from(TOY_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    for edge in graph['edges']:
        rowLabel, colLabel = edge
        adj_matr.at[rowLabel, colLabel] = 1
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# part c - take a look (RUN THIS CELL AS IS)
TOY_ADJ_MATR = get_adj_matr(TOY_GRAPH)
print(TOY_ADJ_MATR)

# COMMAND ----------

# part d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################
    for edge in graph['edges']:
      node = edge[0]
      adj_list[node].append(edge[1])
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# part d - take a look (RUN THIS CELL AS IS)
TOY_ADJ_LIST = get_adj_list(TOY_GRAPH)
print(TOY_ADJ_LIST)

# COMMAND ----------

# MAGIC %md # Question 3: Markov Chains and Random Walks
# MAGIC 
# MAGIC As you know from your readings and in class discussions, the PageRank algorithm takes advantage of the machinery of Markov Chains to compute the relative importance of a webpage using the hyperlink structure of the web (we'll refer to this as the 'web-graph'). A Markov Chain is a discrete-time stochastic process. The stochastic matrix has a principal left eigen vector corresponding to its largest eigen value which is one. A Markov chain's probability distribution over its states may be viewed as a probability vector. This steady state probability for a state is the PageRank of the corresponding webpage. In this question we'll briefly discuss a few concepts that are key to understanding the math behind PageRank. 
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?
# MAGIC 
# MAGIC * __b) short response:__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC 
# MAGIC * __c) short response:__ A Markov chain consists of $n$ states plus an $n\times n$ transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the $n$ states? what implications does this have about the size of the transition matrix?
# MAGIC 
# MAGIC * __d) code + short response:__ What is a "right stochastic matrix"? Fill in the code below to compute the transition matrix for the toy graph from question 2. [__`HINT:`__ _It should be right stochastic. Using numpy this calculation can be done in one line of code._]
# MAGIC 
# MAGIC * __e) code + short response:__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. How many iterations does it take to converge? Which node is most 'central' (i.e. highest ranked)? Does this match your intuition? 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.

# COMMAND ----------

# MAGIC %md ### Q3 Student Answers:
# MAGIC > __a)__ The PageRank metric is the steady state probability distribution of the Markov process underlying the random surfer model of web navigation. Essentially PageRank is a link analysis algorithm that "measures” relative importance of each page within the webgraph.
# MAGIC 
# MAGIC > __b)__ If the conditional probability distribution of future states of a stochastic process (conditional on both past and present values) depends only upon the present state; that is, if we know the present, the future behaviour of the process does not depend on the past, then this property is called Markov property. Example: Brownian motion. P(future | present, past) = P(future| present) 
# MAGIC 
# MAGIC > __c)__ Each state represents a web-page where our random surfer can land. The $n$ states are the number of web-pages, estimated in the order of billions (10^9). A $n x n$ transition matrix would be huge, with a number of entries in the order of 10^18!
# MAGIC 
# MAGIC > __d)__ A stochastic matrix is a square matrix of non-negative real numbers in a closed interval that list the probabilities in a finite Markov chain. This is also called a probability matrix. A right stochastic matrix has each row summing to 1. To transform the adjacency matrix to a right stochastic matrix, we will need to divide every value in the adjacency matrix that is 1 by the row sum.
# MAGIC 
# MAGIC > __e)__ It takes 13 iteractions to converge. Node 'E' has the highest rank. We expected the rank of B,C,D and E to be the same as they have the same in-degree.

# COMMAND ----------

# part d - recall what the adjacency matrix looked like (RUN THIS CELL AS IS)
TOY_ADJ_MATR

# COMMAND ----------

# part d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
transition_matrix = TOY_ADJ_MATR/np.array(TOY_ADJ_MATR).sum(axis=1, keepdims=True)
################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# part e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing inial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = None
    ################ YOUR CODE HERE #################
    state_vector = xInit
    
    for ix in range(nIter):
      
      new_state_vector = state_vector@tMatrix
      state_vector = new_state_vector
      
      if verbose:
        print(f'Step {ix+1}: {state_vector}')
    
    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# part e - run 10 steps of the power_iteration (RUN THIS CELL AS IS)
xInit = np.array([1.0, 0, 0, 0, 0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix, 15, verbose = True)

# COMMAND ----------

# MAGIC %md __`Expected Output for part e:`__  
# MAGIC >Steady State Probabilities:
# MAGIC ```
# MAGIC Node A: 0.10526316  
# MAGIC Node B: 0.15789474  
# MAGIC Node C: 0.18421053  
# MAGIC Node D: 0.23684211  
# MAGIC Node E: 0.31578947  
# MAGIC ```

# COMMAND ----------

# MAGIC %md # Question 4: Page Rank Theory
# MAGIC 
# MAGIC Seems easy right? Unfortunately applying this power iteration method directly to the web-graph actually runs into a few problems. In this question we'll tease apart what we meant by a 'nice graph' in Question 3 and highlight key modifications we'll have to make to the web-graph when performing PageRank. To start, we'll look at what goes wrong when we try to repeat our strategy from question 3 on a 'not nice' graph.
# MAGIC 
# MAGIC __`Additional References:`__ http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) code + short response:__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]
# MAGIC 
# MAGIC * __b) short response:__  Identify the dangling node in this 'not nice' graph and explain how this node causes the problem you described in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __d) short response:__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __e) short response:__ What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.

# COMMAND ----------

# MAGIC %md ### Q4 Student Answers:
# MAGIC > __a)__ The sum of the probabilities is not 1 as it would be in a "nice" graph. We are losing probability mass in each iteraction. This happens because node E has no out-links.
# MAGIC 
# MAGIC > __b)__ Node 'E' is the dangling node in this graph. The adjacency list for E is empty. Any probability mass passed to it will get out of the system. One fix would be to redistribute the mass that goes to 'E' evenly to all the nodes in the graph (including 'E' itself). This is equivalent to set the 'E' row in the matrix with entry values of 1/5.
# MAGIC 
# MAGIC > __c)__ A graph is called irreducible if for every pair i,j of nodes there is a path from i to j and from j to i, that is, we can reach all nodes independent of the node we start from. A web-graph is not naturally irreducible. The inherent assumption for irreducibility is that all web-pages are connected to each other, which is not a realistic assumption. 
# MAGIC 
# MAGIC > __d)__  A graph is aperiodic if the GCD (greatest common divisor) of all cycle lengths is 1. The GCD is also called its period. The web graph is not aperiodic naturally. 
# MAGIC 
# MAGIC > __e)__ An easy fix for reducibility would be to replace the row corresponding to the dangling node 'E' with a column vector with all entries 1/#nodes. This is equivalent of assigning probability of 1/#pages to all pages when transitioning from the the 'problematic' page. An fix for periodicity is to introduce a single self loop. As the number of iterations approaches infinity, the probabilities converge. PageRank introduces a random jump factor that adds a small amount of probability that our random surfer will teleport to any other page instead of randomly follow a hyperlink from its current page to the next as usual. This teleportation factor solves at once both issues of periodicity and irreducibility.

# COMMAND ----------

# part a - run this code to create a second toy graph (RUN THIS CELL AS IS)
TOY2_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
              'edges':[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), 
                       ('B', 'E'), ('C', 'A'), ('C', 'E'), ('D', 'B')]}

# COMMAND ----------

# part a - simple visualization of our test graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY2_GRAPH['nodes'])
G.add_edges_from(TOY2_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################
TOY2_ADJ_MATR = get_adj_matr(TOY2_GRAPH)
transition_matrix = np.nan_to_num(TOY2_ADJ_MATR.div(TOY2_ADJ_MATR.sum(axis=1), axis=0)) 
xInit = np.array([1.0, 0, 0, 0, 0])
power_iteration(xInit, transition_matrix, 10, verbose = True)
################ (END) YOUR CODE #################

# COMMAND ----------

# MAGIC %md # About the Data
# MAGIC The main dataset for this data consists of a subset of a 500GB dataset released by AWS in 2009. The data includes the source and metadata for all of the Wikimedia wikis. You can read more here: 
# MAGIC > https://aws.amazon.com/blogs/aws/new-public-data-set-wikipedia-xml-data. 
# MAGIC 
# MAGIC As in previous homeworks we'll be using a 2GB subset of this data, which is available to you in this dropbox folder: 
# MAGIC > https://www.dropbox.com/sh/2c0k5adwz36lkcw/AAAAKsjQfF9uHfv-X9mCqr9wa?dl=0. 
# MAGIC 
# MAGIC Use the cells below to download the wikipedia data and a test file for use in developing your PageRank implementation (note that we'll use the 'indexed out' version of the graph) and to take a look at the files.

# COMMAND ----------

dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# open test_graph.txt file to see format (RUN THIS CELL AS IS)
with open('/dbfs/mnt/mids-w261/HW5/test_graph.txt', 'r') as f_read:
  for line in f_read:
    print(line)

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
# DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH + 'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + 'indices.txt')
wikiRDD = sc.textFile(DATA_PATH + 'all-pages-indexed-out.txt')

# COMMAND ----------

# display testRDD (RUN THIS CELL AS IS)
testRDD.take(10)

# COMMAND ----------

# display indexRDD (RUN THIS CELL AS IS)
indexRDD.take(10)

# COMMAND ----------

# display wikiRDD (RUN THIS CELL AS IS)
wikiRDD.take(10)

# COMMAND ----------

# MAGIC %md # Question 5: EDA part 1 (number of nodes)
# MAGIC 
# MAGIC As usual, before we dive in to the main analysis, we'll peform some exploratory data analysis to understand our dataset. Please use the test graph that you downloaded to test all your code before running the full dataset.
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) short response:__ In what format is the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC 
# MAGIC * __b) code + short response:__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC 
# MAGIC * __c) code:__ In the space provided below write a Spark job to count the _total number_ of nodes in this graph. 
# MAGIC 
# MAGIC * __d) short response:__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]

# COMMAND ----------

# MAGIC %md ### Q5 Student Answers:
# MAGIC > __a)__ The graph data is encoded as an adjacency list, where the first value (the key) represents a wiki ID, while the value (a hash table) represents the set of wiki neighbors of that specific wiki ID (i.e., hyperlinks in the source wiki ID pointing to other wiki IDs), with an associated payload representing the number of occurrences. For example, in `2\t{'3': 1}` we have the information that wiki '2' is linked to wiki '3' by an hyperlink that occurs only 1 time. 
# MAGIC 
# MAGIC > __b)__ Dangling nodes (i.e., nodes without outgoing links) are not represented as keys in the adjacency lists leading to an undercount of the total number of nodes. 
# MAGIC 
# MAGIC > __d)__ There are `15,192,277 - 5,781,290 = 9,410,987` dangling nodes in the graph.

# COMMAND ----------

# part b - count the number of records in the raw data (RUN THIS CELL AS IS)
# 5781290
print(wikiRDD.count())

# COMMAND ----------

# part c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############
    
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    # parse the data
    parsedRDD = dataRDD.map(parse) \
                       .cache()
    
    # broadcast the set of non-dangling nodes
    ndNodes = sc.broadcast(set(parsedRDD.map(lambda x: x[0]).collect()))
    
    # get the set of dangling nodes by traversing the edges
    dNodes = set(parsedRDD.flatMap(lambda x: [dNode for dNode in x[1] if dNode not in ndNodes.value]).collect())
    
    # sum the counts
    totalCount = len(ndNodes.value) + len(dNodes)
    
    ############## (END) YOUR CODE ###############   
    return totalCount

# COMMAND ----------

# part c - run your counting job on the test file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(testRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part c - run your counting job on the full file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(wikiRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# MAGIC %md # Question 6 - EDA part 2 (out-degree distribution)
# MAGIC 
# MAGIC As you've seen in previous homeworks the computational complexity of an implementation depends not only on the number of records in the original dataset but also on the number of records we create and shuffle in our intermediate representation of the data. The number of intermediate records required to update PageRank is related to the number of edges in the graph. In this question you'll compute the average number of hyperlinks on each page in this data and visualize a distribution for these counts (the out-degree of the nodes). 
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ In the space provided below write a Spark job to stream over the data and compute all of the following information:
# MAGIC  * count the out-degree of each non-dangling node and return the names of the top 10 pages with the most hyperlinks
# MAGIC  * find the average out-degree for all non-dangling nodes in the graph
# MAGIC  * take a 1000 point sample of these out-degree counts and plot a histogram of the result. 
# MAGIC  
# MAGIC  
# MAGIC * __b) short response:__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others?
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ At each iteraction of the PageRank algorithm, the current PageRank mass of the origin node is distributed to all of its out degree's nodes, in proportion to the number of directed connections (e.g. weight) among them.
# MAGIC 
# MAGIC > __c)__ A node with zero out-degree (i.e., a dangling node) does not have any directed edge (or hiperlink) connecting it to another node. At each interaction of the algorithm, we will redistribute the PageRank mass of all dangling nodes evenly across all the nodes in the graph, in order to conserve the total PageRank mass constant.

# COMMAND ----------

# part a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############

    # count the out-degree of each non-dangling node
    outdegreeRDD = dataRDD.map(parse) \
                          .mapValues(lambda x: len(x.keys())) \
                          .cache()
    
    # get the 10 nodes with most edges
    top = outdegreeRDD.takeOrdered(10, key=lambda x: -x[1])
    
    # average out-degree for all non-dangling nodes
    avgDegree = outdegreeRDD.map(lambda x: x[1]) \
                            .mean()
    
    # take a n size sample with non-replacement of the out-degree counts
    fraction = n/outdegreeRDD.count()
    sampledCounts = outdegreeRDD.sample(False, fraction, seed=100) \
                                .map(lambda x: x[1]) \
                                .collect()
    
    ############## (END) YOUR CODE ###############
    
    return top, avgDegree, sampledCounts

# COMMAND ----------

# part a - run your job on the test file (RUN THIS CELL AS IS)
start = time.time()
test_results = count_degree(testRDD,10)
print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", test_results[1])
print("Top 10 nodes (by out-degree:)\n", test_results[0])

# COMMAND ----------

# part a - plot results from test file (RUN THIS CELL AS IS)
plt.hist(test_results[2], bins=10)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# part a - run your job on the full file (RUN THIS CELL AS IS)
start = time.time()
full_results = count_degree(wikiRDD,1000)

print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", full_results[1])
print("Top 10 nodes (by out-degree:)\n", full_results[0])

# COMMAND ----------

# part a - plot results from full file (RUN THIS CELL AS IS)
plt.hist(full_results[2], bins=50)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# MAGIC %md # Question 7 - PageRank part 1 (Initialize the Graph)
# MAGIC 
# MAGIC One of the challenges of performing distributed graph computation is that you must pass the entire graph structure through each iteration of your algorithm. As usual, we seek to design our computation so that as much work as possible can be done using the contents of a single record. In the case of PageRank, we'll need each record to include a node, its list of neighbors and its (current) rank. In this question you'll initialize the graph by creating a record for each dangling node and by setting the initial rank to 1/N for all nodes. 
# MAGIC 
# MAGIC __`NOTE:`__ Your solution should _not_ hard code \\(N\\).
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) short response:__ What is \\(N\\)? Use the analogy of the infinite random web-surfer to explain why we'll initialize each node's rank to \\(\frac{1}{N}\\). (i.e. what is the probabilistic interpretation of this choice?)
# MAGIC 
# MAGIC * __b) short response:__ Will it be more efficient to compute \\(N\\) before initializing records for each dangling node or after? Explain your reasoning.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code below to create a Spark job that:
# MAGIC   * parses each input record
# MAGIC   * creates a new record for any dangling nodes and sets it list of neighbors to be an empty set
# MAGIC   * initializes a rank of 1/N for each node
# MAGIC   * returns a pair RDD with records in the format specified by the docstring
# MAGIC 
# MAGIC 
# MAGIC * __d) code:__ Run the provided code to confirm that your job in `part a` has a record for each node and that your should records match the format specified in the docstring and the count should match what you computed in question 5. [__`TIP:`__ _you might want to take a moment to write out what the expected output should be fore the test graph, this will help you know your code works as expected_]
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q7 Student Answers:
# MAGIC 
# MAGIC > __a)__ \\(N\\) is the total number of nodes (i.e., wikis or pages) in the graph. At the beginning, there is a 1/\\(N\\) probability of a random web-surfer landing in any particular page.
# MAGIC 
# MAGIC > __b)__ It's more efficient to initialize the records for each dangling nodes first, append them as new rows in our adjacency list, and then perform a total count of rows in order to recover \\(N\\). 

# COMMAND ----------

# part c - job to initialize the graph (RUN THIS CELL AS IS)
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############

    # write any helper functions here
    
    def parse(line):
      node, edges = line.split('\t')
      return (node, ast.literal_eval(edges))
    
    # write your main Spark code here
    
    # parse the data
    parsedRDD = dataRDD.map(parse) \
                       .cache()
    
    # broadcast the set of non-dangling nodes
    ndNodes = sc.broadcast(set(parsedRDD.map(lambda x: x[0]).collect()))
    
    # initialize the dangling nodes RDD by traversing the edges
    dNodesRDD = parsedRDD.flatMap(lambda x: [dNode for dNode in x[1] if dNode not in ndNodes.value]) \
                         .distinct() \
                         .map(lambda x: (x, {})) \
                         .cache()
    
    # count total number of nodes
    N = len(ndNodes.value) + dNodesRDD.count()
    
    # merge RDDs and emit records with initial PageRank   
    graphRDD = parsedRDD.union(dNodesRDD) \
                        .mapValues(lambda x: (1/N, x))
    
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# COMMAND ----------

# part c - run your Spark job on the test graph (RUN THIS CELL AS IS)
start = time.time()
testGraph = initGraph(testRDD).collect()
print(f'... test graph initialized in {time.time() - start} seconds.')
testGraph

# COMMAND ----------

# part c - run your code on the main graph (RUN THIS CELL AS IS)
start = time.time()
wikiGraphRDD = initGraph(wikiRDD).cache()
print(f'... full graph initialized in {time.time() - start} seconds')

# COMMAND ----------

# part c - confirm record format and count (RUN THIS CELL AS IS)
start = time.time()
print(f'Total number of records: {wikiGraphRDD.count()}')
print(f'First record: {wikiGraphRDD.take(1)}')
print(f'... initialization continued: {time.time() - start} seconds')

# COMMAND ----------

# MAGIC %md # Question 8 - PageRank part 2 (Iterate until convergence)
# MAGIC 
# MAGIC Finally we're ready to compute the page rank. In this last question you'll write a Spark job that iterates over the initialized graph updating each nodes score until it reaches a convergence threshold. The diagram below gives a visual overview of the process using a 5 node toy graph. Pay particular attention to what happens to the dangling mass at each iteration.
# MAGIC 
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/PR-illustrated.png?raw=true' width=50%>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __`A Note about Notation:`__ The formula above describes how to compute the updated page rank for a node in the graph. The $P$ on the left hand side of the equation is the new score, and the $P$ on the right hand side of the equation represents the accumulated mass that was re-distributed from all of that node's in-links. Finally, $|G|$ is the number of nodes in the graph (which we've elsewhere refered to as $N$).
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: $\alpha * \frac{1}{|G|}$
# MAGIC 
# MAGIC * __b) short response:__ In the equation for the PageRank calculation above what does $m$ represent and why do we divide it by $|G|$?
# MAGIC 
# MAGIC * __c) short response:__ Keeping track of the total probability mass after each update is a good way to confirm that your algorithm is on track. How much should the total mass be after each iteration?
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code below to create a Spark job that take the initialized graph as its input then iterates over the graph and for each pass:
# MAGIC   * reads in each record and redistributes the node's current score to each of its neighbors
# MAGIC   * uses an accumulator to add up the dangling node mass and redistribute it among all the nodes. (_Don't forget to reset this accumulator after each iteration!_)
# MAGIC   * uses an accumulator to keep track of the total mass being redistributed.( _This is just for your own check, its not part of the PageRank calculation. Don't forget to reset this accumulator after each iteration._)
# MAGIC   * aggregates these partial scores for each node
# MAGIC   * applies telportation and damping factors as described in the formula above.
# MAGIC   * combine all of the above to compute the PageRank as described by the formula above.
# MAGIC   * 
# MAGIC   
# MAGIC    __WARNING:__ Some pages contain multiple hyperlinks to the same destination, please take this into account when redistributing the mass.
# MAGIC 
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q8 Student Answers:
# MAGIC 
# MAGIC > __a)__ In a given proportion of times (represented by the alpha factor), instead of randomly clicking in a hyperlink as usual, the random web surfer is teleported to a new page. There is a 1/|G| chance of landing at any particular page, where |G| is the number of nodes in the graph. 
# MAGIC 
# MAGIC > __b)__ m represents the total mass coming from the dangling nodes. We divide it by |G| (i.e., the count of nodes) to evenly split it among all nodes in our graph. 
# MAGIC 
# MAGIC > __c)__ The total probability mass should always equal to precision 1. 

# COMMAND ----------

# part d - provided FloatAccumulator class (RUN THIS CELL AS IS)

from pyspark.accumulators import AccumulatorParam

class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.

    def MapDNodes(line):
      '''Helper function to process the dangling nodes
      and update dangling and total mass accumulators.
      Extended to all nodes only due to totAccum.
      Returns:
            None
      '''      
      # update accumulators
      mass = line[1][0]
      if len(line[1][1])>0:
        totAccum.add(mass)
      else:
        mmAccum.add(mass)
        totAccum.add(mass)      
    
    def MapALLNodes(line):
      '''Helper function to process all nodes to emit mass
      to neighbors (if neighbors exist) + graph structure.
      Returns:
            emit_list - list of KV pairs [(node, (mass, edges))]
      '''
      # initialize emit_list
      emit_list = []

      # pass over the graph structure
      emit_list.append((line[0], (float(0.0), line[1][1])))

      # emit mass to neighbors (if neighbors exist)
      mass = line[1][0]
      neighbors = line[1][1]
      connections = sum(neighbors.values())
      for n in neighbors:
        emit_list.append((n, (neighbors[n]*mass/connections, {})))

      for newline in emit_list:
        yield newline
          
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace
    
    # broadcast count of nodes
    G = sc.broadcast(graphInitRDD.count())
    
    # split the graphInitRDD in two: graphStructRDD (partitioned by key, cached, immutable) and steadyStateRDD
    graphStructRDD = graphInitRDD.mapValues(lambda x: x[1]).partitionBy(6).cache()
    steadyStateRDD = graphInitRDD.mapValues(lambda x: x[0]).cache()
    
    # loop maxIter times
    for ix in range(maxIter):
      
      # join RDDs
      joinedRDD = steadyStateRDD.join(graphStructRDD) \
                                .cache()
      
      # process dangling nodes
      joinedRDD.foreach(MapDNodes)
      
      # broadcast dangling nodes accumulated mass
      dm = sc.broadcast(mmAccum.value)
      
      # process all nodes, recover mass and data structure, and update scores by aplying the PR formula
      steadyStateRDD = joinedRDD.flatMap(MapALLNodes) \
                                .reduceByKey(lambda x,y: (x[0]+y[0], {**x[1],**y[1]})) \
                                .mapValues(lambda x: a.value/G.value + d.value*(dm.value/G.value+x[0])) \
                                .cache()

      # print checks (if verbose == True)
      if verbose:
        print('-'*50, 'INTERACTION', ix+1, '-'*50)
        print('Total distributed mass:', round(totAccum.value, 6))
        print('Top-10 nodes:', steadyStateRDD.takeOrdered(10, key=lambda x: -x[1]))

      # reset accumulators
      mmAccum.value = 0
      totAccum.value = 0
    
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

# COMMAND ----------

# part d - run PageRank on the test graph (RUN THIS CELL AS IS)
# NOTE: while developing your code you may want turn on the verbose option
nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = False)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
test_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md __`expected results for the test graph:`__
# MAGIC ```
# MAGIC [(2, 0.3620640495978871),
# MAGIC  (3, 0.333992700474142),
# MAGIC  (5, 0.08506399429624555),
# MAGIC  (4, 0.06030963508473455),
# MAGIC  (1, 0.04255740809817991),
# MAGIC  (6, 0.03138662354831139),
# MAGIC  (8, 0.01692511778009981),
# MAGIC  (10, 0.01692511778009981),
# MAGIC  (7, 0.01692511778009981),
# MAGIC  (9, 0.01692511778009981),
# MAGIC  (11, 0.01692511778009981)]
# MAGIC ```

# COMMAND ----------

# part d - run PageRank on the full graph (RUN THIS CELL AS IS)
# NOTE: wikiGraphRDD should have been computed & cached above!
nIter = 10
start = time.time()
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = False)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

top_20 = full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# Save the top_20 results to disc for use later. So you don't have to rerun everything if you restart the cluster.
top_20_df = spark.createDataFrame(top_20, ['nodeID', 'pageRank'])
top_20_df.write.parquet(f"{blob_url}/top_20_HW05")

# COMMAND ----------

# view record from indexRDD (RUN THIS CELL AS IS)
# title\t indx\t inDeg\t outDeg
indexRDD.take(1)

# COMMAND ----------

# map indexRDD to new format (index, name) (RUN THIS CELL AS IS)
namesKV_RDD = indexRDD.map(lambda x: (str(x.split('\t')[1]), x.split('\t')[0])).cache()

# COMMAND ----------

# see new format (RUN THIS CELL AS IS)
namesKV_RDD.take(2)

# COMMAND ----------

# MAGIC %md # OPTIONAL
# MAGIC ### The rest of this notebook is optional and doesn't count toward your grade.
# MAGIC The indexRDD we created earlier from the indices.txt file contains the titles of the pages and their IDs.
# MAGIC 
# MAGIC * __a) code:__ Join this dataset with your top 20 results.
# MAGIC * __b) code:__ Print the results

# COMMAND ----------

# MAGIC %md ## Join with indexRDD and print pretty

# COMMAND ----------

# read parquet from blob storage and cast it as RDD
top_20_df = spark.read.parquet(f"{blob_url}/top_20_HW05/*")
top_20_rdd = top_20_df.rdd

# COMMAND ----------

# part a
joinedWithNames = None
############## YOUR CODE HERE ###############
joinedWithNames = namesKV_RDD.join(top_20_rdd).collect()
############## END YOUR CODE ###############

# COMMAND ----------

# part b
# Feel free to modify this cell to suit your implementation, but please keep the formatting and sort order.
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in joinedWithNames:
    print ("{:6f}\t| {:10d}\t| {}".format(r[1][1],int(r[0]),r[1][0]))

# COMMAND ----------

# MAGIC %md ## OPTIONAL - GraphFrames
# MAGIC GraphFrames is a graph library which is built on top of the Spark DataFrames API.
# MAGIC 
# MAGIC * __a) code:__ Using the same dataset, run the graphframes implementation of pagerank.
# MAGIC * __b) code:__ Join the top 20 results with indices.txt and display in the same format as above.
# MAGIC * __c) short answer:__ Compare your results with the results from graphframes.
# MAGIC 
# MAGIC __NOTE:__ Feel free to create as many code cells as you need. Code should be clear and concise - do not include your scratch work. Comment your code if it's not self annotating.

# COMMAND ----------

# MAGIC %md ### Student Answers:
# MAGIC 
# MAGIC > __c)__ xxx

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from graphframes import *
from pyspark.sql import functions as F

# COMMAND ----------

# spark initialization
sc = spark.sparkContext

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC ### You will need to generate vertices (v) and edges (e) to feed into the graph below. 
# MAGIC Use as many cells as you need for this task.

# COMMAND ----------

def getVertexEdges(dataRDD):
    """
    Spark job to read in the raw data and extract two 
    DFs: the vertexDF and the edgesDF.
    
    Returns: 
        vertexDF -  DF of (node_id, node_name)
        edgesDF - DF of (source_id, destination_id)
    """

    # parse the data
    parsedRDD = dataRDD.map(lambda x: (x.split('\t')[0], ast.literal_eval(x.split('\t')[1]))) \
                       .cache()

    # broadcast the set of non-dangling nodes
    ndNodes = sc.broadcast(set(parsedRDD.map(lambda x: x[0]).collect()))

    # get the set of dangling nodes by traversing the edges
    dNodes = set(parsedRDD.flatMap(lambda x: [dNode for dNode in x[1] if dNode not in ndNodes.value]).collect())
  
    # store the vertex in a DF
    vertexDF = sc.parallelize(ndNodes.value.union(dNodes)).map(lambda x: (int(x), x)).toDF(['id', 'name'])
  
    # store the edges in a DF
    edgesDF = parsedRDD.flatMap(lambda x: [(x[0], dst, x[1][dst]) for dst in x[1]]) \
                       .flatMap(lambda x: x[2]*[(int(x[0]), int(x[1]), 'follow')]) \
                       .toDF(['src', 'dst', 'relationship"'])

    return vertexDF, edgesDF

# COMMAND ----------

# Create a GraphFrame
v, e = getVertexEdges(wikiRDD)
g = GraphFrame(v, e).cache()

# COMMAND ----------

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.15, maxIter=10)

# COMMAND ----------

# MAGIC %md
# MAGIC As discussed in Vini's OH our group and others are having trouble in running the pageRank method of graphframes in the cluster. We tried several times, but have run into memory or executor failure issues.

# COMMAND ----------

start = time.time()
top_20 = results.vertices.orderBy(F.desc("pagerank")).limit(20)
print(f'... completed job in {time.time() - start} seconds.')

# COMMAND ----------

# MAGIC %%time
# MAGIC top_20.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Run the cells below to join the results of the graphframes pagerank algorithm with the names of the nodes.

# COMMAND ----------

namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

namesKV_DF = namesKV_RDD.toDF()

# COMMAND ----------

namesKV_DF = namesKV_DF.withColumnRenamed('_1','id')
namesKV_DF = namesKV_DF.withColumnRenamed('_2','title')
namesKV_DF.take(1)

# COMMAND ----------

resultsWithNames = namesKV_DF.join(top_20, namesKV_DF.id==top_20.id).orderBy(F.desc("pagerank")).collect()

# COMMAND ----------

# TODO: use f' for string formatting
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in resultsWithNames:
    print ("{:6f}\t| {:10s}\t| {}".format(r[3],r[2],r[1]))

# COMMAND ----------

# MAGIC %md ### Congratulations, you have completed HW5! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform
