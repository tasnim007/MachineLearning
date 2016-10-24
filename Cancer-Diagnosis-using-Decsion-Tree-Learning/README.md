# Diagnosis of cancer using Decision Tree Learning

## Problem Description:

In this mini-project, we implemented a decision-tree algorithm and apply it to breast cancer diagnosis. For each patient, an image of a fine needle aspirate (FNA) of a breast mass was taken, and nine features in the image potentially correlated with breast cancer were extracted. Our task is to develop a decision tree algorithm, learn from data, and predict for new patients whether they have breast cancer. Dataset was downloaded from U.C. Irvine Machine Learning Repository.

1.	Each patient is represented by one line, with columns separated by commas: the first one is the identifier number, the last is the class (benign or malignant), the rest are attribute values, which are integers ranging from 1 to 10. The attributes are: Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses. 

2.	Implementation of the ID3 decision tree Algorithm.

3.	Implementation of both misclassification impurity and information gain for evaluation criterion. Also, implementation of split stopping using chi-square test.

4.	Divide the data set randomly between training (80%) and testing (20%) sets. The implemented algorithm was used to train a decision tree classifier and report accuracy on test. Run the same experiment 100 times. Then calculate average test performances (accuracy, precision, recall, f-measure, g-mean).





## Implementation:
**Language:** C++
