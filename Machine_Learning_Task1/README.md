This task was long but TL;DR summary:

Problem 1:
Some theoretical ml questions, did not bother including here.

Problem 2: Regression:
Predict water temperature from salinity using polynomial regression (degrees 1–6), then try degree 4 with L2 regularization. Analyze results.

a. Polynomial regression on first 700 points, degrees 1–6; plot data + regression curves and cost vs degree.
b. Fix degree 4, add L2 regularization with various λ; plot data + regression curves and cost vs λ.

Problem 3: k-NN:
Classify iris flowers using k-NN. First use 2 features and k=3, then vary k, finally use all features. Compare results.

a. Split iris data, k-NN with k=3 on first 2 features; print accuracy, plot training data + decision regions.
b. Vary k from 1 to 15 (still 2 features); plot accuracy vs k, find best k.
c. Use all features, repeat b; compare results.

Problem 4: Text & Naive Bayes:
Classify tweets as disaster/non-disaster using Naive Bayes. Analyze most common words and compute word importance (LR metric).

a. Clean data, create features, split 80:20, train Multinomial Naive Bayes, achieve ≥65% test accuracy.
b. Find 5 most frequent words in positive and negative tweets; compute and analyze LR metric for words appearing ≥10 times in both classes.
