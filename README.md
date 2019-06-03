# Evaluate your word embeddings!

#### Summary:
* [Introduction](#introduction)
* [Installation](#installation)
    * [Requirements](#requirements)
    * [Clone and install](#clone-and-install)
    * [Download benchmarks](#download-benchmarks)
* [Description of evaluation methods](#description-of-evaluation-methods)
    * [Similarity and relatedness](#similarity-and-relatedness)
    * [Feature-norms](#feature-norms)
    * [Concreteness](#concreteness)
* [Quick examples](#quick-examples)
* [Disclaimer](#disclaimer)

## Introduction
This repository implements popular methods to evaluate word embeddings. It is meant to be easy to use to perform quick analysis of the quality of word embeddings.

## Installation

### Requirements
* linux
* python
* scipy
* numpy
* pickle
* sklearn
* json

### Clone and install
* Clone the repository `git clone https://github.com/EloiZ/embedding_evaluation.git`
* Add these lines to your `~/.bashrc`:
    * `export PYTHONPATH=$PYTHONPATH:path/to/embedding_evaluation/`
    * `export EMBEDDING_EVALUATION_DATA_PATH='path/to/embedding_evaluation/data/'`

### Download benchmarks
* To download the benchmarks, go to the root dir of the repo and run:
    * `$./embedding_evaluation/download_benchmarks.py`

## Description of evaluation methods

### Similarity and relatedness

Semantic  relatedness  (similarity)  evaluates  the  similarity  (relatedness)  degree  of  word  pairs.  Several  benchmarks provide gold  labels  (i.e.  human   judgment   scores)   for   word   pairs. The spearman correlation is computed between  the  list  of  similarity  scores  given  by the  embeddings  (cosine-similarity  between  vectors) and the gold labels. The higher the correlation is, the more semantic is captured in the embeddings. While word similarity benchmarks are widely used for intrinsic embedding evaluation, they are biased in the sense that good intrinsic evaluation  scores  do  not  imply  useful  embeddings  for downstream tasks as shown by [9]. However, this is a standard intrinsic evaluation method, used for example in [10], [11], [12]. Benchmarks supported are

* MEN [1]
* WordSim353 [2]
* SimLex-999 [3]
* USF [6]
* VisSim and SemSim [4]

### Feature-norms



Given word representation, try to predict to feature norms (e.g. given "apple" and its embedding, predict "is_round", "is_eadable", "is_fruit"...). There is a total of 43 characteristics grouped into 9 categories for 417 entities. A linear SVM classifier is trained and 5-fold validation scores are reported. This method is used for example in [10], [13]. Benchmark supported:

* McRae feature norms [5]

### Concreteness
Evaluate how well the word embeddings contain concreteness information. A SVM with a RBF kernel to predict the gold concreteness rating from word embeddings. This method is used for example in [10]. Benchmark supported: 

* USF concreteness [6]

## Quick examples
The global approach is given with those simple lines:
```python
from embedding_evaluation.evaluate import Evaluation
from embedding_evaluation.load_embedding import load_embedding_textfile

# Load embeddings as a dictionnary {word: embed} where embed is a 1-d numpy array.
embeddings = load_embedding_textfile(textfile_path="path/to/my/embeddings.csv")

# Load and process evaluation benchmarks
evaluation = Evaluation() 

results = evaluation.evaluate(embeddings)
```
`results` contains a dictionary with all evaluation scores.
Depending on the format of your embeddings, you can use different embedding loader:
#### csv file
Your embeddings are in a csv file that looks like this:
```
cat,0.71,0.64,0.67,-0.23,...
dog,-0.20,0.11,1.72,-0.89,...
house,...
...
```
Then you can load the embeddings with `embeddings = load_embedding_textfile(textfile_path="path/to/my/embeddings.csv")`

#### npy file
Your embeddings are in a `.npy` file that contains the matrix with shape `N X D` where `N` is the vocabulary size and `D` the dimension of the embeddings. You need a vocabulary file with the same order as the `.npy` file that contains the list of the words, for example:
```
cat
dog
house
```
Then you can load the embeddings with `embeddings = load_embedding_npy(npy_path="path/to/my/embeddings.npy", vocab_path="path/to/vocab.txt")`

## Disclaimer
This is on-going work and any kind of feedback (bugs, code optimization, feature requests...) is more than welcome :) Do not hesitate to contribute and/or to open an issue.


References
==========

* [1] Bruni, E., Tran, N., and Baroni, M. (2014). Multimodal distributional semantics. <em>J. Artif. Intell. Res.</em>, 49:1–47.
* [2] Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z., Wolfman, G., and Ruppin, E. (2002). Placing search in context: the concept revisited. <em>ACM Trans. Inf. Syst.</em>, 20(1):116–131.
* [3] Hill, F., Reichart, R., and Korhonen, A. (2015). Simlex-999: Evaluating semantic models with (genuine) similarity estimation. <em>Computational Linguistics</em>, 41(4):665–695.
* [4] Silberer, C. and Lapata, M. (2014). Learning grounded meaning representations with autoencoders. <em>In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics, ACL 2014, June 22-27, 2014, Baltimore, MD, USA, Volume 1: Long Papers,</em> pages 721–732.
* [5] McRae, K., Cree, G. S., Seidenberg, M. S., and McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. <em>Behavior research methods</em>, 37(4):547–559.
* [6] Nelson,   D.   L.,   McEvoy,   C.   L.,   and Schreiber,  T.  A.  (2004).   The  university  of  south  florida  free association, rhyme, and word fragment norms. <em>Behavior Research Methods, Instruments, & Computers</em>, 36(3):402–407.
* [9] Faruqui, M., Tsvetkov, Y., Rastogi, P., and Dyer, C. (2016). Problems with evaluation of word embeddings using word similarity tasks. <em>arXiv preprint arXiv:1605.02276</em>
* [10] Zablocki, É., Piwowarski, B., Soulier, L., and Gallinari, P. (2018). Learning Multi-Modal Word Representation Grounded in Visual Context. <em>In Association for the Advancement of Artificial Intelligence (AAAI), New Orleans, United States.</em>
* [11] Collell, G., Zhang, T., and Moens, M. (2017). Imagined visual representations as multimodal embeddings. <em>In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, February 4-9, 2017, San Francisco, California, USA.</em>, pages 4378–4384.
* [12] Lazaridou, A., Pham, N. T., and Baroni, M. (2015). Combining language and vision with a multimodal skip-gram model. <em>In NAACL HLT 2015, The 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Denver, Colorado, USA, May 31 - June 5, 2015</em>, pages 153–163.
* [13] Collell, G. and Moens, M. (2016). Is an image worth more than a thousand words? on the fine-grain semantic differences between visual and linguistic representations. <em>In COLING 2016, 26th International Conference on Computational Linguistics, Proceedings of the Conference: Technical Papers, December 11-16, 2016, Osaka, Japan,</em> pages 2807–2817.
