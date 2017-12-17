# Performance
#### 100 features + 5-NN (original)

```
Evaluator
Accuracy: 0.363073110285
Precision: 0.504438747594
Recall: 0.346058237005
Fscore: 0.410501593376
Confusion matrix:
[[93 12 13  0  0  0  0  0]
 [56 51  0  8  0  0  0  1]
 [52  1 48  0  0  0  0  0]
 [49 10  0 11  3  0  0  3]
 [22  4  5  2 59  0  0  2]
 [72 16 23  2  0  1  0  0]
 [47  5  4  4 14  1  2  3]
 [37  4 11  7 20  0  1 28]]
Done in 112.333334923 secs.
```

## Assessing the descriptor
### SIFT + SVM
####  50 features

```
Evaluator
/home/jon/repos/mcv/m3/scene-classificator/.venv/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
Accuracy: 0.094175960347
  'precision', 'predicted', average, warn_for)
Precision: 0.0117719950434
Recall: 0.125
Fscore: 0.0215175537939
Confusion matrix:
[[  0   0   0 118   0   0   0   0]
 [  0   0   0 116   0   0   0   0]
 [  0   0   0 101   0   0   0   0]
 [  0   0   0  76   0   0   0   0]
 [  0   0   0  94   0   0   0   0]
 [  0   0   0 114   0   0   0   0]
 [  0   0   0  80   0   0   0   0]
 [  0   0   0 108   0   0   0   0]]
Done in 46.7203950882 secs.
```

#### 100 features

```
Evaluator
/home/jon/repos/mcv/m3/scene-classificator/.venv/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
Accuracy: 0.193308550186
Precision: 0.110470501521
Recall: 0.17665830688
Fscore: 0.135935727715
Confusion matrix:
[[  0  85  33   0   0   0   0   0]
 [  0 106  10   0   0   0   0   0]
 [  0  57  44   0   0   0   0   0]
 [  0  65  11   0   0   0   0   0]
 [  0  53  35   0   6   0   0   0]
 [  0  96  18   0   0   0   0   0]
 [  0  54  26   0   0   0   0   0]
 [  0  58  44   0   6   0   0   0]]
Done in 82.7260429859 secs.
```

####  200 features

```
Accuracy: 0.148698884758
Precision: 0.0350543478261
Recall: 0.12920075979
Fscore: 0.0551465149401
Confusion matrix:
[[  6 112   0   0   0   0   0   0]
 [  1 114   0   0   0   0   1   0]
 [ 30  71   0   0   0   0   0   0]
 [  0  76   0   0   0   0   0   0]
 [  0  94   0   0   0   0   0   0]
 [  3 111   0   0   0   0   0   0]
 [  0  80   0   0   0   0   0   0]
 [  6 102   0   0   0   0   0   0]]
/home/jon/repos/mcv/m3/scene-classificator/.venv/lib/python2.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
Done in 186.497986078 secs.
```

### Colour histogram + SVM
#### 256 bins
```
Evaluator
Accuracy: 0.17843866171
Precision: 0.186824009989
Recall: 0.17545726468
Fscore: 0.180962318846
Confusion matrix:
[[52  5 10  9 30  1  4  7]
 [29  3 13 12 32  3 16  8]
 [38  6 15  1 35  0  3  3]
 [21  2 14  8 15  1  4 11]
 [31  0  9  5 29  3 13  4]
 [36 10 10 10 12  7  5 24]
 [26  2  6  9 22  2 11  2]
 [29  6 12 15 19  3  5 19]]
Done in 21.1547560692 secs.

```

#### 128 bins
```
Evaluator
Accuracy: 0.235439900867
Precision: 0.25937878362
Recall: 0.234597601065
Fscore: 0.246366596829
Confusion matrix:
[[58  2  6 10 12  4  9 17]
 [31 14  8 11 12  4 23 13]
 [34  1 25  4 14  1 16  6]
 [15  5 11 14  5  0 14 12]
 [32  1 14 10  9  2 25  1]
 [36 16 12  8  2 17  8 15]
 [22  2 11  4  9  1 30  1]
 [27  8 12 15  4  4 15 23]]
Done in 11.4477789402 secs.

```

#### 64 bins
```
Evaluator
Accuracy: 0.234200743494
Precision: 0.245853322064
Recall: 0.236887184943
Fscore: 0.241286987635
Confusion matrix:
[[49  6  4  5 10  6 22 16]
 [29 12  3  8 12 15 30  7]
 [23  4 24  3 15  1 27  4]
 [22  2  6 10  5  5 19  7]
 [24  1 16  2  7  5 37  2]
 [37 12  5  6  6 25  8 15]
 [17  1  6  5  5  1 43  2]
 [17  8 10 11  2  7 34 19]]
Done in 9.82988405228 secs.

```

#### 32 bins
```
Evaluator
Accuracy: 0.263940520446
Precision: 0.27084657817
Recall: 0.265390879005
Fscore: 0.268090975348
Confusion matrix:
[[49  2  3  2  9 12 24 17]
 [29  5  5  3 10 25 33  6]
 [15  1 32  1  8  6 31  7]
 [11  0  8  6  7 17 20  7]
 [21  0  9  5  9  8 38  4]
 [24  3  3  2  5 53 13 11]
 [11  0  8  3  3  5 50  0]
 [19  1 12  9  2 21 35  9]]
Done in 2.47506904602 secs.
```

#### 16 bins
```
Evaluator
Accuracy: 0.262701363073
Precision: 0.22275806829
Recall: 0.263715097957
Fscore: 0.241512461018
Confusion matrix:
[[45  3  4  2  9 17 29  9]
 [26  1  6  2  6 33 36  6]
 [13  5 33  1  4  7 32  6]
 [ 9  2 12  6  5 14 25  3]
 [18  1 12  4  2 10 45  2]
 [18  1  2  1  3 68 15  6]
 [10  0  9  1  2  6 52  0]
 [18  1  9  4  1 27 43  5]]
Done in 2.23412513733 secs.
```

#### 8 bins
```
Evaluator
Accuracy: 0.251548946716
Precision: 0.226733807491
Recall: 0.253642610802
Fscore: 0.239434546323
Confusion matrix:
[[41  3  3  0  5 21 34 11]
 [22  2  4  2  4 38 39  5]
 [14  4 30  1  2 10 34  6]
 [ 8  1  6  5  4 21 28  3]
 [15  0 14  4  1 10 47  3]
 [20  2  4  0  0 66 14  8]
 [ 8  0  9  0  1  7 54  1]
 [13  0  9  4  1 26 51  4]]
Done in 2.4412419796 secs.
```


### Best escriptor: Colour Histogram with 32 bins

#### [BEST_DESCRIPTOR] + KNN 5
```
Evaluator
Accuracy: 0.195786864932
Precision: 0.2344837194
Recall: 0.192255962154
Fscore: 0.211280530175
Confusion matrix:
[[46 32  8  4 15  0  2 11]
 [19 27 14 15 20  0  9 12]
 [14 41 24  1 19  0  2  0]
 [15  8 11  6 20  0  2 14]
 [25 21 19  3 14  1  9  2]
 [39 17 17 10 11  2  4 14]
 [16 12  6  4 18  1 22  1]
 [23 15 17 13 21  1  1 17]]
Done in 17.8198068142 secs.
```

#### [BEST_DESCRIPTOR] + RandomForest
```
Evaluator
Accuracy: 0.220570012392
Precision: 0.22398942925
Recall: 0.223529886629
Fscore: 0.223759421995
Confusion matrix:
[[23 21 19  7 10 14  9 15]
 [24 21 20  6  8 18  5 14]
 [17  6 41  8  9  3  6 11]
 [16 14  5 15  5  3 10  8]
 [21  8 15  7 11  6 21  5]
 [28 19 12  7  6 26  1 15]
 [16  7  7 10  9  2 26  3]
 [20 19  9 14 10  8 13 15]]
Done in 1.25663304329 secs.

```

#### [BEST_DESCRIPTOR] + GaussianBayes
```
Evaluator
Accuracy: 0.246592317224
Precision: 0.250558141859
Recall: 0.245034679465
Fscore: 0.24776563072
Confusion matrix:
[[28 18 11 19 13 14  4 11]
 [11 31  4 21 11 23  1 14]
 [12  6 39  2 12  7 10 13]
 [ 9 14  8 13  0 12  8 12]
 [23  7 11  9  8 14 12 10]
 [17 22  5 10  9 37  1 13]
 [15  1  7 15  3  9 28  2]
 [20 15  1 30  4 12 11 15]]
Done in 1.2053668499 secs.
```

#### [BEST_DESCRIPTOR] + BernoulliBayes
```
Evaluator
Accuracy: 0.251548946716
Precision: 0.21270465026
Recall: 0.247445765284
Fscore: 0.228763739791
Confusion matrix:
[[19 27 19  4  2 18 14 15]
 [ 9 36 14  0  7 33  7 10]
 [ 7  8 55  1  1  4 18  7]
 [15 18  8  2  2  9 19  3]
 [19 12  9  4  2  9 30  9]
 [ 8 38 10  2  1 44  3  8]
 [12  5  2 10  3  7 35  6]
 [13 34  3  6  4 22 16 10]]
Done in 1.19816017151 secs.

```

#### [BEST_DESCRIPTOR] + SVM
```
Evaluator
Accuracy: 0.263940520446
Precision: 0.27084657817
Recall: 0.265390879005
Fscore: 0.268090975348
Confusion matrix:
[[49  2  3  2  9 12 24 17]
 [29  5  5  3 10 25 33  6]
 [15  1 32  1  8  6 31  7]
 [11  0  8  6  7 17 20  7]
 [21  0  9  5  9  8 38  4]
 [24  3  3  2  5 53 13 11]
 [11  0  8  3  3  5 50  0]
 [19  1 12  9  2 21 35  9]]
Done in 2.47506904602 secs.
```

#### [BEST_DESCRIPTOR] + LogisticRegression
```
Evaluator
Accuracy: 0.161090458488
Precision: 0.167387600448
Recall: 0.149596518182
Fscore: 0.157992787286
Confusion matrix:
[[  0  10   3   0   0 102   3   0]
 [  0   4   1   0   0 110   0   1]
 [  0  12   7   0   0  78   4   0]
 [  1   7   3   1   0  61   2   1]
 [  0   3   9   0   0  76   5   1]
 [  0   7   0   0   0 106   1   0]
 [  0   8   6   1   1  52  12   0]
 [  0   7   6   0   0  93   2   0]]
Done in 3.05640506744 secs.
```


## Best classifier: [BEST_CLASSIFIER

## Assessing the best classifier with the best descriptor

###