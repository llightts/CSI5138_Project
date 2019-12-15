# CSI 5138 Project - WiC SuperGlue Task with Sense Fine-Tuning

### Task
This repo contains the Python 3 Jupyter Notebooks and the datasets used to compete in the SuperGlue WiC task (https://pilehvar.github.io/wic/). The goal of this research project is to compare a Baseline RoBERTa base model against a RoBERTa base model that has been fine-tuned with a Sense-disambiguation head. Both the Baseline and the Sense fine-tuned models also contain a WiC head which is trained on the WiC training set. By improving upon the Baseline's Contextual Word Embeddings through fine-tuning we will exhibit that the computationally feasible fine-tuning approach to modifying a BERT-based model's Contextual Word Embeddings is an effective method of achieving high, state-of-the-art results. Compared to the sense pre-training task implemented in SenseBERT, this sense fine-tuning task is more efficient and provides the behaviour that we hypothesize will occur.

### Notes
- The following Jupyter Notebooks contain the code used for every step of the project (data preprocessing, data loading, the models, the training, and the testing). These are messy implementations that will need to be converted into a clean library once more research is performed.
- The SemCor and SensEval datasets are used for training the Sense disambiguation head that we later use to modify the RoBERTa Contextual Word Embeddings. The WiC training set is used to then train the WiC classifier and update the RoBERTa model accordingly.
- We currently test on the WiC validation set due to the WiC test set's labels not being provided to the public. Thus, the findings are based on comparisons between the third best performing model on the task (RoBERTa) and our RoBERTa-based architecture (adding the sense-fine tuning head) 
- More tests will be needed to achieve more conclusive statistical testing

### Findings
- There has been an average of a 1.2% increase of our implementation compared to the RoBERTa Baseline model.
- With 85% confidence (on four tests), there is a statistically significant difference between the two models from anlyzing the confidence intervals and by performing the paired t-test between the models. This indicates that our implementation is probably statistically better at the WiC task than the Baseline mode.
- From comparing the incorrect results obtained from the two models, we find that our implementation is better at determining when a word is used in a different context in two different sentences (better sense disambiguation). But we also find that our Sense training implementation can be improved with more experimentation since our model learns to differentiate some words too much (especially when the senses are similar).
