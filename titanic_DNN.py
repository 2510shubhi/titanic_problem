from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import model_selection

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

### fill missing values of sex and Age
sexCheck = lambda s: 1 if s == 'male' else 0
train_data['Sex'] = train_data['Sex'].apply(sexCheck)
### fill missing age values
mean_age = train_data['Age'].mean()
train_data.fillna(mean_age, inplace=True)

titanic_features = ['Parch','Pclass', 'Sex', 'Age']
titanic_target = ['Survived']
X = train_data[titanic_features]
Y = train_data[titanic_target]

# Splitting the data using sklearn model_selection
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.2)
clf = MLPClassifier(hidden_layer_sizes = (10), random_state = 1)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_val)
print ("Train set score: %f" % clf.score(X_train, Y_train))
print ("Test set score: %f" % clf.score(X_val, Y_val))

#### on test data
sexCheck = lambda s: 1 if s == 'male' else 0
test_data['Sex'] = test_data['Sex'].apply(sexCheck)

# Inputing missing age values
mean_test_age = test_data['Age'].mean()
test_data.fillna(mean_test_age, inplace=True)

#make precitions
X_test = test_data[titanic_features]
predictions=clf.predict(X_test).astype(int)
print(predictions)
# convert to csv format
result_df=pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
result_df.to_csv("result_dnn.csv", index=False)
