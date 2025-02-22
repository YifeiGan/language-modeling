# language-modeling
HW1 Report

Yifei Gan

1 Introduction

The primary objective is to evaluate the model’s ability to predict tokens in a sequence, compute probabilities for various contexts, and assess performance using perplexity, a standard metric in language modeling that quantifies the model’s uncertainty in predicting the next token. Lower perplexity scores indicate better model perfor- mance and reflect the model’s understanding of the underlying language structure. This project focuses on the problem of unsupervised language modeling, which involves training a model to predict the probability of a token in a sequence given the preceding context. Specifically, the task is to develop an autoregressive language model capable of predicting the next token in a sequence of text. The goal is to estimate the conditional probabilities of tokens p(ti|t<i), where ti is the current token, and t<i represents the tokens preceding it. The model’s performance will be evaluated using perplexity, a widely used metric in language modeling that measures the model’s uncertainty in predicting the next token.

The dataset provided is the Penn Treebank (PTB) corpus, a standard benchmark dataset for natural language processing tasks. PTB contains approximately 1 million tokens, split across training, validation, and test sets. The corpus consists of cleaned and tokenized text, making it suitable for sequence modeling tasks.

Dataset Split Sample Sentence![](Aspose.Words.91bb0f3a-d7bc-4414-9046-c2c3921e8f06.001.png)![](Aspose.Words.91bb0f3a-d7bc-4414-9046-c2c3921e8f06.002.png)

Training Set pierre <unk> N years old will

join the board as a nonexecutive director nov. N

Validation Set consumers may want to move their

telephones a little closer to the tv set

Test Set no it was n’t black monday![](Aspose.Words.91bb0f3a-d7bc-4414-9046-c2c3921e8f06.003.png)

Table 1: Penn Treebank Dataset Sample

Training Set: Contains the majority of the tokens ( 930,000 tokens).

Validation Set: Used for hyperparameter tuning ( 74,000 tokens).

Test Set: Used for final evaluation ( 82,000 tokens). The input takes sequences of tokens (e.g., words or subwords) from the PTB dataset, and outputs the conditional probability p(ti|t<i) of each token in the sequence.

2 Model

For the models presented in this paper, an embed- ding layer is used to convert the input tokens into dense vectors. The dimensionality of these embed- dings is set to 100. The embeddings are trained from scratch, allowing the model to learn represen- tations tailored specifically for the Penn Treebank dataset. I opted for trainable embeddings instead of pre-trained embeddings (e.g., GloVe or Word2Vec) to ensure that the model’s vocabulary aligns per- fectly with the dataset used, reducing the risk of out-of-vocabulary issues. This approach is particu- larly beneficial for small datasets like PTB, where training embeddings with the model can capture specific relationships unique to the dataset.

The primary model used in this study is a Long Short-Term Memory (LSTM) network, which is well-suited for capturing long-range dependencies in sequences. The LSTM model is implemented with an embedding layer, followed by an LSTM layer with 256 hidden units, and a fully connected output layer. The model is trained using the Penn Treebank dataset, with cross-entropy loss as the objective function. Key hyperparameters include: Embedding Size: 100

Hidden Size: 256

Batch Size: 32

Sequence Length: 30

Learning Rate: 0.001 (Adam optimizer)

Training was conducted over 5 epochs with gradi- ent clipping to stabilize the training process and

prevent exploding gradients. The hidden state is detached between batches to avoid backpropagat- ing through the entire sequence history, which can lead to memory issues and hinder convergence. The LSTM model was chosen due to its ability to address the vanishing gradient problem encoun- tered in traditional RNNs. The Adam optimizer was used to ensure efficient training with adaptive learning rates.

3 Experiments

The training set was used to optimize the model parameters, while the validation set was used to select the best hyperparameters and avoid overfit- ting. The test set was reserved for final evaluation to assess the model’s generalization performance. To determine the best hyperparameters for the LSTM model, I utilized Optuna, an open-source hyperparameter optimization framework. The ob- jective function for Optuna was defined based on minimizing both cross-entropy loss and perplexity on the validation set. Optuna performed a series of trials to search for the optimal combination of hyperparameters, including learning rate, embed- ding size, hidden size, and batch size. The final set of hyperparameters was selected based on the trial that achieved the lowest validation loss and perplexity.

The PTB dataset is relatively small, and as a result, the model is at risk of overfitting. To mitigate this, I used regularization techniques such as dropout within the LSTM layers. Furthermore, the <unk> token was used to handle out-of-vocabulary words, allowing the model to generalize better to unseen data. To address potential data imbalance issues, we ensured that the training, validation, and test splits maintained consistent distributions of sen- tence lengths and vocabulary. FortheLSTMmodel, differentvaluesforthehyper- parameters were tested during the Optuna tuning process:

Learning Rate: Tested values ranged from 0.0001 to 0.01. The best learning rate was found to be 0.001, balancing fast convergence and stability. Hidden Size: We experimented with hidden sizes ranging from 128 to 512 units. The final model used 256 hidden units, providing a good trade-off between model capacity and computational effi- ciency.

Embedding Size: We tested embedding sizes of 50, 100, and 150. An embedding size of 100 was cho-

sen based on performance and the model’s ability tocapturemeaningfulrelationshipsbetweenwords. During training, I used the Adam optimizer with gradient clipping set to 5. This prevented the gradi- ents from becoming too large, which could desta- bilize training, especially in RNN-based models. Dropout was applied to the LSTM layers to prevent overfitting and improve generalization.

The primary metric used to evaluate model perfor- mance was perplexity, which measures how well the model predicts a sequence of words. Perplex- ity is the exponential of the average negative log- likelihood of predicted tokens, with lower values indicating better performance. Cross-entropy loss was also used to evaluate the models during train- ing.

To select the final model for generating the test set submissions, I focused on the model that achieved the lowest validation perplexity. This ensured that the selected model had the best capability to gen- eralize to new, unseen text. The results were vali- dated using the test set, and the perplexity scores for individual sentences were logged to analyze the variability in model performance across different sentence structures.

4 Results

The performance of the LSTM model was evaluated on the train, validation, and test sets using perplexity as the primary metric. Different hyperparameter choices had a significant impact on model performance. For instance, increasing the hidden size improved the model’s capacity to learn complex relationships, but beyond 256 hidden units, the model started to overfit, as indicated by increasing validation perplexity. Similarly, the learning rate played a crucial role in convergence; rates lower than 0.001 resulted in slower convergence, while higher rates led to instability during training. The choice of embedding size also impacted performance. An embedding size of 100 provided a good balance between model capacity and training efficiency. Embedding sizes smaller than 100 resulted in insufficient representational power, whereas larger sizes increased computational cost without a corresponding improvement in performance.

The best-performing model was the LSTM with 256 hidden units, an embedding size of 100, a batch size of 32, and a learning rate of 0.001. This configuration achieved the lowest validation

perplexity and balanced both model complexity and training efficiency. The use of Optuna for hyperparameter tuning helped identify this optimal set of hyperparameters efficiently.

The following table compares the validation and test perplexity for the modified model versus the regular model:



|Model|Validation Perplexity|Test Perplexity|
| - | - | - |
|Modified Model Regular Model|180\.3928 243.7412|166\.2182 222.3850|

Table 2: Comparison of Validation and Test Perplexity between Modified and Regular Models

The modified model achieved significantly lower perplexity compared to the regular model, indicat- ing better generalization and reduced overfitting. Previous work has also shown that LSTMs with

similar hidden sizes are effective for language mod- eling tasks.

![](Aspose.Words.91bb0f3a-d7bc-4414-9046-c2c3921e8f06.004.png)

Figure 1: Perplexity change

5 Reference

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a Large Annotated Corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313–330.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. Neural Com- put. 9, 8 (November 15, 1997), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
