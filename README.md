# Machine Learning Real Life Examples
Machine learning real life examples made by me. These include coding examples and explanations on the problems being addressed + the model results from the dataset.

## Table of Contents
* [Regression](#Regression)
   * [Problem 1](#problem-1)
   * [Problem 2](#problem-2)
   * [Problem 3](#problem-3)
* [Classification](#Classification)
   * [Problem 1](#problem-1)
* [Clustering](#Clustering)
   * [Problem 1](#problem-1)
* [Dimensionality Reduction](#Dimensionality-Reduction)
   * [Problem 1](#problem-1)

## Regression

### Problem 1
This consists of a very simple dataset that has two columns: Years' Experience & Salary. We've been hired as a data scientist to find out the correlations between the Salary and Years' Experience.

Using a simple linear regression we can find the best fit for an employee's salary depending on their years of experience.

Upon training our simple linear regression model we can predict the outcome of our models results. Using the y_pred variable and comparing it against y_test we can see that the predictions are very accurate. The code can be found [here](https://github.com/Achronus/ML-Real-Life-Examples/blob/master/code-examples-and-data-files/code-examples/simple_linear_regression.py)

![YPred vs YTest](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/ypred-vs-ytest.png)

Using matplot we can create graphs to view the training set and test set results.

![Simple Linear Regression Graphs](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/simple-linear-graphs.png)

As you can see from the training set, the predictions vary (the blue line) compared to the real values that the model has been trained on (the red dots).

However, when viewing the test set (the new observations) against the predictions you can see that they are very accurate.


### Problem 2
This consists of a dataset with five columns: R&D Spend, Administration, Marketing Spend, State & Profit. We've been hired as a data scientist by a venture capitalist fund company to analyse the dataset to provide them with information on which companies they would be most interested in investing in. Their main criteria is the profit.

Using a small sample of their data, they want to determine where companies perform better based on the state they are in through the use of Marketing Spend, Administration or R&D Spend.

Using a multiple linear regression we can help them to identify this.

Using backward elimination (a feature selection method from dimensionality reduction) we can reduce the models independent variables. This allows us to identify the best columns that impact the companies performance based on their profit. The code can be found [here](https://github.com/Achronus/ML-Real-Life-Examples/blob/master/code-examples-and-data-files/code-examples/multiple-linear-regression.py)

![R&D Spend vs Marketing Spend](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/rndspend-vs-marketingspend.png)

Using a significance level of 0.05, the results show that the R&D Spend is the most important factor relating to the companies profit. However, Even though the Marketing Spend is 0.06 this is arguably still an important factor relating to the companies profit. Comparing the R-squared and Adj. R-squared when having both columns in place, the values are higher than when only having the R&D Spend.

In conclusion, R&D Spend and Marketing Spend are the two most important factors when considering the companies profit.


### Problem 3
We are a part of the human resources department in a large company and are looking to hire a new employee. We have found the ideal candidate and are considering making an offer to them. However, we are unsure on the appropriate salary for their position.

The new employee has told us that he has 20+ years' experience and has earned over 160k salary in his previous company of employment and is asking for this as a starting wage. Doing research on the previous company we identified ten positions and the salary per position.

Based on the position data we have, we have determined that the new employee is around a level of 6.5 and want to be accurate with the salary. Looking at the data we can see that there is a non-linear relationship and we need to put a model together to determine the new employee's starting salary. The code can be found [here](https://github.com/Achronus/ML-Real-Life-Examples/blob/master/code-examples-and-data-files/code-examples/polynomial_regression.py)

In order to complete this problem we are going to use a polynomial regression.

Using the graph below you can clearly identify the correct price for each employee level and can accurately predict the position level 6.5.

![Polynomial Regression Graph](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/poly-reg-graph.png)


## Classification

### Problem 1
We are hired as a data scientist for a social media business that has a client for a car company that wants to use their ad services to promote a new SUV. Our job is to create a model that identifies which users have bought the SUV through the social network.

Based on the dataset provided, we are going to use the Age and Estimated Salary columns to help us with this problem. The code can be found [here](https://github.com/Achronus/ML-Real-Life-Examples/blob/master/code-examples-and-data-files/code-examples/naive_bayes.py)

In order to solve this we are going to use the Naïve Bayes model.

![Naive Bayes Graph](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/naive-bayes-graph.png)

As you can see from the graphs above, there is a larger ratio of customers that did not purchase the SUV. 1 being purchased (green), 0 being that the customer didn't purchase the SUV (red).


## Clustering

### Problem 1
We have been hired as a machine learning specialist to work for a mall that has provided us with some of their customer's information to a membership card that they offer to customers to purchase products in different shops. The dataset provided consists of: Customer ID, Gender, Age, Annual Income (k$) & Spending Score (1-100).

Our job is to create a model that segregates the customers into respective groups based on their Annual Income & Spending Score. The code can be found [here](https://github.com/Achronus/ML-Real-Life-Examples/blob/master/code-examples-and-data-files/code-examples/hc.py)

We are going to address this problem using Hierarchical Clustering.

![Hierarchical Clustering Graph](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/hierarchical-clustering-graph.png)

From viewing the dendrogram you can see that that 5 clusters is the optimal number of groups to use for the dataset. From the graph, I have labelled each cluster to determine the type of customers that are within the dataset.

The groups consist of: Careful - these are the people who have a high income but do not spend often; Standard - medium income and spend quite often; Target - have a high budget that spend quite often, the customers to go for; Careless - very small income but spend often; Sensible - low income and don't spend often.

## Dimensionality Reduction

### Problem 1
A wine business owner has collated information related to types of wines they offer to customers. We have been hired as a data scientist to identify the best independent variables for each customer segment to determine which segment is suitable for each wine type. The code can be found [here](https://github.com/Achronus/ML-Real-Life-Examples/blob/master/code-examples-and-data-files/code-examples/lda.py)

As the dataset is so large, we are going to us the feature extraction model LDA to solve this issue.

![LDA Graph](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/real-life-examples/lda-graph.png)

Using a logistic regression combined with the LDA, the graphs show you a clean split between each of the three wine types.
