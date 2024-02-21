# graph-contrastive-learning

In this project, we aim to benchmark different graph contrastive learning algorithms on biomedical  networks targeting a range of different prediction tasks.

The main idea is to have something that resembles:
> [6] Yue et al. Graph embedding on biomedical networks: methods, applications and evaluations. Bioinformatics, 2019.

Various representation learning approaches have been introduced to deal with prediction and classification tasks on graphs.

> [1] Thomas N. Kipf, Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR, 2017

> [2] W.L. Hamilton, R. Ying, and J. Leskovec. Inductive Representation Learning on Large Graphs. In NeurIPS, 2017.

Characteristic examples include shallow architectures (e.g., DeepWalk, Node2Vec, EFGE) and Graph Neural Networks (GNNs) models such as Graph Convolutional Networks and Graph Attention Networks.

> [3] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. Graph Attention Networks. In ICLR, 2018.
> [4] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu. A Comprehensive Survey on Graph Neural Networks. arXiv, 2019.

Nevertheless, GNN models primarily focus on (semi-)supervised learning tasks that requires access to annotated (labeled) data. 
To address this issue, recent research efforts in graph self-supervised learning (SSL) have focused on generating graph 
representations independently of annotated data. Among different approaches, graph contrastive learning has gain significant 
attention due to the comparable performance to supervised approaches in various graph representation learning tasks.

> [7] Liu et al. Graph Self-Supervised Learning: A Survey. IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.
> [8] Zhu et al. An Empirical Study of Graph Contrastive Learning. In NeurIPS 2021


Course from January on:

### Forward propagation
 - Aggregation
Aggregation information from neighbors
simplest fucntion can be a sum. it can be scaled. invariant function
aggregation can be a neural network itself

- Transformations
Once aggregated we need to transform information
    - add edges
    - remove edges
    - change vector of info of node

The forward propagation happens among the nodes of the graph.
We need to have an adequate number of layer (related to the diameter of the graph).

In most of the papers 3 layers -> it is a hyperparameter to fit
Over smoothing when too much layers because aggregation

- Markov chain
We can consider aggregation to a random walk and if too many layers it will converge to a sttionary distribution


### Learning

- **Unsupervised learning**, basic tools to infer

- **Supervised learning**, requires access to labeled data

- **Semi-supervised learning**, label can be infered from neighbors

- **Self-supervised learning**, learning from labeled data but not rely on labeled data

- **Contrastive learning**, apply transformation to data and force model to understand them as close to each other (ie: compute similare embeddings)



Models must create embeddings with positive and negative exampels
We might want to constrast the nodes or contrast full graphs

When representing node in embedding a 'pooling' function should compute the embedding of the graph


# To do

### First
Build on  top of python package (most intensive part):
> [8] Zhu et al. An Empirical Study of Graph Contrastive Learning. In NeurIPS 2021

### Then
Biomedical prediciton task from:
> [5] Biomedical datasets: https://github.com/xiangyue9607/BioNEV  

> [6] Yue et al. Graph embedding on biomedical networks: methods, applications and evaluations. Bioinformatics, 2019.
Basically the same as our objective but they consider less interesting algorithms.




### Finally
Benchmark / Pipeline that allow to test and compare.  
Under which circumstances a model is more suited to data ?


### Summary

1. Read about GNN  
Stanford course on Youtube by J Leskovec machine learning with graphs and websites at stanford with materials

2. Take a look at those that discribe widely used models:

> [1] Thomas N. Kipf, Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR, 2017

> [2] W.L. Hamilton, R. Ying, and J. Leskovec. Inductive Representation Learning on Large Graphs. In NeurIPS, 2017

> [3] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. Graph Attention Networks. In ICLR, 2018.

Essentially:

> [4] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu. A Comprehensive Survey on Graph Neural Networks. arXiv, 2019.

3. Start to read about self-supervised:

> [7] Liu et al. Graph Self-Supervised Learning: A Survey. IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.  

> [8] Zhu et al. An Empirical Study of Graph Contrastive Learning. In NeurIPS 2021  

proposes abstraction that were shown in the slides of presentation

4. Build code above:
> [9] https://github.com/PyGCL/PyGCL



Invest time on code  
GBL Deepmind  
Meeting every 2 weeks until January then more intensive


# Follow-up

- [x] GCN 
- [x] GAT
- [x] GraphSAGE
- :white_check_mark: test