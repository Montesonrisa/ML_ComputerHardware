import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


dane = pd.read_csv('ComputerHardware.csv')
parameters = dane[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX']]
performance = dane[['PRP']]
print(parameters.head(2))
print(performance.tail(2))

X_train, X_test, y_train, y_test = train_test_split(parameters, performance, test_size=0.2)
#Linear regression model based on sklearn.linear_model.LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions=model.predict(X_test)
result = pd.DataFrame()
result ['actual']=  y_test['PRP']
result ['predicted']=  model.predict(X_test)

"""
Comparative analysis of the article authors model 
and the linear regression model based on sklearn.linear_model.LinearRegression
"""

plt.figure()
plt.scatter(dane['PRP'], dane['ERP'],  marker='^', color='r')
plt.scatter(result['actual'], result['predicted'], color='b')
plt.title('Comparison of models')
plt.xlabel('published relative performance')
plt.ylabel('estimated relative performance')
plt.legend(['model from the article', 'the model based on ML'])
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.show()

#Model based on sklearn.tree.DecisionTreeRegressor
reg = DecisionTreeRegressor ()
tree = reg.fit(X_train, y_train)
print (tree)

#R^2 analysis sklearn.linear_model.LinearRegression
R2_test=r2_score(y_test, predictions)
R2_train=r2_score(y_train, model.predict(X_train))
print("sklearn.linear_model.LinearRegression: ")
print ("R^2 score for test set: ", R2_test)
print ("R^2 score for train set: ", R2_train)

#R^2 analysis sklearn.tree.DecisionTreeRegressor
R2_treetest=r2_score(y_test, predictions)
R2_treetrain=r2_score(y_train, model.predict(X_train))
print("sklearn.tree.DecisionTreeRegressor: ")
print ("R^2 score for test set: ", R2_treetest)
print ("R^2 score for train set: ", R2_treetrain)

if R2_test>R2_treetest:
    r2=R2_test
else: r2=R2_treetest

if r2> 0.9:
    print ("Goodness of fit is very high")
elif r2> 0.8:
    print ("Goodness of fit is high")
elif r2 > 0.8:
    print("Goodness of fit is high")
elif r2 > 0.6:
    print("Goodness of fit is satisfactory")
elif r2 > 0.5:
    print("Goodness of fit is low")
else:
    print("Goodness of fit is unsatisfactory")


print(100*'.')

#R^2 analysis of model from article
R2=r2_score(dane['PRP'], dane['ERP'])
print("The estimation made by the authors of the article")
print ("R^2 score: ", R2)

if R2>r2:
    print("The estimation made by the authors of the article fits better than estimation made with Machine Learning")
elif R2==r2:
    print("Goodness of fit is same for both estimations")
else: print("The estimation made by the authors of the article fits worse than estimation made with Machine Learning")