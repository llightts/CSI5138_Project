# CSI 5138 Project - WiC SuperGlue Task with Sense Fine-Tuning
**By: Simon Fortier-Garceau, Julian Templeton and William Larocque**

As part of the course CSI5138F: Intro to DL and RL by Pr. Yongyi Mao at the University of Ottawa.
## Description
### Task
This repo contains the Python 3 Jupyter Notebooks and the datasets used to compete in the [SuperGlue](https://super.gluebenchmark.com/) [WiC task](https://pilehvar.github.io/wic/). The goal of this research project is to compare a Baseline RoBERTa base model against a RoBERTa base model that has been fine-tuned with a Sense-disambiguation head. Both the Baseline and the Sense fine-tuned models also contain a WiC head which is trained on the WiC training set. By improving upon the Baseline's Contextual Word Embeddings through fine-tuning we will exhibit that the computationally feasible fine-tuning approach to modifying a BERT-based model's Contextual Word Embeddings is an effective method of achieving high, state-of-the-art results. Compared to the sense pre-training task implemented in SenseBERT, this sense fine-tuning task is more efficient and provides the behaviour that we hypothesize will occur.

### Notes
- The following Jupyter Notebooks contain the code used for every step of the project (data preprocessing, data loading, the models, the training, and the testing). These are messy implementations that will need to be converted into a clean library once more research is performed. They were built to run on [Google Colab](https://colab.research.google.com/) with our project drive mounted for IO.
- The SemCor and SensEval datasets are used for training the Sense disambiguation head that we later use to modify the RoBERTa Contextual Word Embeddings. The WiC training set is used to then train the WiC classifier and update the RoBERTa model accordingly.
- We currently test on the WiC validation set due to the WiC test set's labels not being provided to the public. Thus, the findings are based on comparisons between the third best performing model on the task (RoBERTa) and our RoBERTa-based architecture (adding the sense-fine tuning head) 
- More tests will be needed to achieve more conclusive statistical testing

### Findings
- There has been an average of a 1.2% increase of our implementation compared to the RoBERTa Baseline model.
- With 85% confidence (on four tests), there is a statistically significant difference between the two models from anlyzing the confidence intervals and by performing the paired t-test between the models. This indicates that our implementation is probably statistically better at the WiC task than the Baseline mode.
- From comparing the incorrect results obtained from the two models, we find that our implementation is better at determining when a word is used in a different context in two different sentences (better sense disambiguation). But we also find that our Sense training implementation can be improved with more experimentation since our model learns to differentiate some words too much (especially when the senses are similar).

## Files
### Google Colab Notebooks
#### [SeperateLoss_Pos_Senses.ipynb](SeparateLoss_Pos_Senses.ipynb)
This is the notebook used to preprocess the SemCor and SenseEval datasets as well as fine-tuned the RoBERTa_base model on the Sense differentiation task. We use Cross-Entropy Loss for the Sense task in this file.

To perform this, we create a special head to go from the RoBERTa embeddings size (768) to the sense vocabulary size (31). Note that the training method still uses the Masked LM task and a POS recognition task. However, we found that they did not help the final results. As such, we set the alpha hyperparameter for both theses models to 0.0, set the POS head to None and set the alpha value for sense to 1.0. We then trained for a varying number of epochs (up to 5). We obtained the best results after 2 epochs.

#### [CombinedLoss_Pos_Senses.ipynb](CombinedLoss_Pos_Senses.ipynb)
A copy of [SeperateLoss_Pos_Senses.ipynb](SeparateLoss_Pos_Senses.ipynb) that uses the same loss funtion as Masked LM for the Sense differentiation and POS recognition tasks.

#### [RoBERTa_WiC_baseline.ipynb](RoBERTa_WiC_baseline.ipynb)
In this file, we load the WiC dataset. We then create a custom head to go from the embeddings (768 dimensions) to 2. The input for this custom head is the difference of the embeddings for both versions of the target words. We use matrix multiplication to extract the embeddings of the target words from the RoBERTa sentence embedding.

This version of the head includes a reference to the RoBERTa model. This means that when we train the it, the RoBERTa embeddings are fine-tuned on the WiC task. We found this to be the best approach.

#### [RoBERTa_WiC_Testing_FullyFrozen.ipynb](RoBERTa_WiC_Testing_FullyFrozen.ipynb)
This copy of [RoBERTa_WiC_baseline.ipynb](RoBERTa_WiC_baseline.ipynb) includes a version of the WiC head without a reference to the RoBERTa model. Instead, we need to pass the RoBERTa model to the *forward* function. This allows us to completely freeze the latter's weights. We experimentally found this method does not yield good results.

#### [RoBERTa_WiC_baseline_analysis.ipynb](RoBERTa_WiC_baseline_analysis.ipynb)
Another copy of [RoBERTa_WiC_baseline.ipynb](RoBERTa_WiC_baseline.ipynb) which incudes the code used to gather and analyze the results when testing the model's performance on the WiC task. These are informal additions that should be incorporated into the original file in the future. The paired t-test has been informally performed locally, so this would also be formally merged to the original in the future as a simple function.

#### [Alpha values experimentation.xlsx](Alpha values experimentation.xlsx)
The results from all formal testing performed for this project. This contains POS and Masked LM aspects that are no longer discussed due to insignificant results from the experimentation.

### Reprocessed datasets
We may upload our processed versions of the SemCor and SenseEval datasets. If they are not present, note that the functions to do that processing are part of the [SeparateLoss_Pos_Senses notebook](SeparateLoss_Pos_Senses.ipynb).

## Datasets used
### Word-in-Context (WiC)
Each entry in this dataset includes a target word, two sentences and a label (for the training and validation set only). The sentences both use the target word. However, they may use a different meaning of said word. The goal of this task is to determine if both use of the word are for the same meaning or not.

We used the version that comes from the [SuperGLUE benchmark site](https://super.gluebenchmark.com/tasks). The data came in 3 jsonl files, one for the training set, one for the validation set and one for the testing set. This version of the dataset includes the character position of each target word in both sentences of each dataset entry. This is invaluable in our implementation as it allowed us to find the embeddings of each word even if it was split in multiple token before going in the RoBERTa model.

### SemCor and SenseEval
We downloaded the latest versions of each of these datasets from [R. Mihalcea's website](https://web.eecs.umich.edu/~mihalcea/downloads.html). We then processed each of these datasets to go from the xml format to a json format for each sentence. We create a json object for each sentence, we keep a string copy of the sentence, we then create the array of words, lemmatized words, parts-of-speech (POS), Wordnet ids and lexical ids. These are then saved in jsonl files similar to the WiC files.

## References
- J. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Google AI Language. NAACL-HLT, 2019. Available from: https://arxiv.org/abs/1810.04805
- N. Latysheva, Why do we use word embeddings in NLP?, Towards Data Science, September 10th, 2019. Available from: https://towardsdatascience.com/why-do-we-use-embeddings-in-nlp-2f20e1b632d2
- M. Peters et al., Knowledge Enhanced Contextual Word Representations. ArXiv, abs/1909.04164, 2019. Available from: https://arxiv.org/pdf/1909.04164.pdf
- Huggingface, Transformers, huggingface.co, 2019. Available from: https://huggingface.co/transformers/
- R. Mihalcea, Downloads, web.eecs.umich.edu. Available from: https://web.eecs.umich.edu/~mihalcea/downloads.html#sensevalsemcor
- Y. Liu et al., Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692, 2019. Available from: https://arxiv.org/pdf/1907.11692.pdf
- Y. Levine et al., Sensebert: Driving some sense into bert. ArXiv, abs/1908.05646, 2019. Available from https://arxiv.org/pdf/1908.05646.pdf
- A. Wang et al., SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems, ArXiv, abs/1905.00537, 2019. Available from: https://arxiv.org/pdf/1905.00537.pdf
- P. Edmonds and S. Cotton, SENSEVAL-2: overview, Proceedings of SENSEVAL-2 Second International Workshop on Evaluating Word Sense Disambiguation Systems, July 2001.
- B. Snyder and M. Palmer, The English all-words task, SENSEVALl@ACL, 2004. Pages 1-5.
- A. Vaswani et al., Attention is all you need. NIPS'17 Proceedings of the 31st International Conference on Neural Information Processing Systems. December 04 - 09, 2017. Pages 6000-6010.
- Mohammad Taher Pilehvar and Jose Camacho-Collados, WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations, In Proceedings of NAACL 2019 (short), Minneapolis, USA. Available from https://arxiv.org/pdf/1808.09121.pdf
- Mohammad Taher Pilehvar and Jose Camacho-Collados, README, WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations,  2019. Available from https://pilehvar.github.io/wic/package/README.txt
- D. and A. Jorge. 2019. Language modelling makes sense: Propagating representations through wordnet for full-coverage word sense disambiguation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, page forthcoming, Florence, Italy. Association for Computational Linguistics. Available from https://www.aclweb.org/anthology/P19-1569/
