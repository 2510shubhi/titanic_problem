# Titanic_problem_kaggle
1. Download the train.csv and test.csv datasets.

2.The models implemented are:
  Deep neural networks
  Random Forest 
  SVM linear and kernelised
  
3.The approach to the problem is:
 
  1. First analysing the dataset of both training and testing. 
  The training/testing data set has missing values in Age, Sex and Embarkment.
  It is necessary to fill the missing values by mean, mode , median.
  I chose mean as it gives a better average of results over the dataset.
  
  2. Choose the features to use in traing and target values for training data.
    It depends on the survivors based on the class, age, sex, fare and parch.
    
  3. Train the classifier over training dataset and make predictions for test set.
  
  4. Create a dictionary of ['Passengerid','Survived'] and convert to csv files. 
    
  

  
