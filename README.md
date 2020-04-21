# mlpce

***Machine Learning Prediction Confidence Estimation***

[![Build Status](https://travis-ci.org/bmewing/mlpce.svg?branch=master)](https://travis-ci.org/bmewing/mlpce)
[![Maintainability](https://api.codeclimate.com/v1/badges/ae6887700d819adba3f1/maintainability)](https://codeclimate.com/github/bmewing/mlpce/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/ae6887700d819adba3f1/test_coverage)](https://codeclimate.com/github/bmewing/mlpce/test_coverage)
[![PyPi version](https://pypip.in/v/mlpce/badge.png)](https://crate.io/packages/mlpce/)

Let's say you have a cool XGBOOST model that you've built and now 
you're wanting to make predictions with it on new data points - how well 
does your training data cover that model space? In classic statistical 
analysis, especially DOEs, there are many characteristics about the data used
to cover a space that can be considered (e.g. A-, D-, G-, I-optimality). 
I-optimality is the average prediction variance in the design space, that is, 
a measure of how precisely a model built on that data should be able to make
new predictions.

`mlpce` is a Python package which provides an expression of confidence in any
given prediction by using an approximating linear function to calculate the 
standard error of prediction for the new point and comparing it to the same
value for the training data. The approximating linear function can either be 
specified as a string or the module will simply pick a high-order polynomial
model based on the available degrees of freedom in the training data.

## Usage

Consider a dataset picked to be I-Optimal for evaluating a full third-order
response surface model. There are 54 rows and 6 columns. This pandas data frame
can then be passed into the Confidence class where an approximating linear model
will be created and the necessary matrices will be calculated. Now we can pass
in a few new rows to be evaluated.

```python
import pandas as pd
from mlpce import Confidence

pd_x = pd.DataFrame(data=[[-1, -0.5, 0.5, -1, 1, 1], [1, -1, 1, -1, -1, -1], [-0.5, 0.5, 1, -0.5, 0, 1],
                          [0.5, 1, 1, 0.5, -1, -1], [-0.5, 0.5, -0.5, 1, -1, 0.5], [-0.5, 0.5, -1, -0.5, 0.5, 1],
                          [1, 1, -1, -1, -1, 0.5], [1, -1, -1, -0.5, 1, 0.5], [1, 0.5, -1, 1, 0.5, 0],
                          [0, -0.5, 0.5, -0.5, -0.5, 0.5], [1, 1, 1, 1, 1, -0.5], [0.5, 1, -0.5, 0.5, -0.5, 1],
                          [0.5, -0.5, -0.5, -0.5, 0.5, -0.5], [1, -1, 1, -1, 0.5, 1], [-1, 1, 0, 1, 1, 1],
                          [1, 1, 0.5, -1, 1, 1], [-0.5, -0.5, -1, -1, 0.5, -1], [1, -1, -1, 0.5, 1, -1],
                          [0.5, -1, -1, -1, -0.5, -0.5], [-1, -1, 0, -0.5, -1, -1], [1, -0.5, 1, 0.5, 1, 0],
                          [0.5, -1, 0.5, 1, 0, -0.5], [1, 0.5, 0.5, -0.5, -0.5, -0.5], [1, -1, 1, 0.5, -1, 1],
                          [0.5, 0.5, -0.5, -1, 1, -1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0, 1, 1, 1],
                          [-0.5, -0.5, 1, 0.5, -1, -0.5], [-1, 1, 0, -0.5, 1, 0], [1, 1, -0.5, -1, -0.5, -1],
                          [0.5, 0.5, -1, 1, -1, -0.5], [0.5, 1, 1, -1, -1, 0.5], [1, -1, -1, 1, -1, 0.5],
                          [-0.5, -1, -0.5, 0.5, 1, 0], [1, -0.5, -0.5, -1, -1, 1], [-1, -0.5, -1, 1, -0.5, -1],
                          [-1, 1, -1, 1, 0.5, -1], [-0.5, -1, -1, -0.5, -1, 1], [-1, 0, -0.5, -1, -0.5, 0.5],
                          [1, -1, 0.5, -1, 1, -1], [-1, 0.5, -1, -0.5, -1, -1], [1, 1, 1, 1, -1, 1],
                          [1, -1, -0.5, 0.5, -1, -1], [-1, 0.5, 1, 1, -1, -1], [-1, -1, 1, -0.5, 1, -0.5],
                          [-1, -0.5, -1, 0.5, 0, 1], [-1, -1, 1, -1, -1, 1], [-1, 0, 0.5, 1, 1, -1], 
                          [0.5, 1, 1, -1, 0.5, -1], [-0.5, 0.5, 1, -1, -1, -1], [-1, 0, 1, 1, -1, 1], 
                          [-1, 1, 0.5, -0.5, -1, 1], [-0.5, 1, 0.5, 0.5, 0, -0.5], [-1, -1, 1, 1, 0.5, 0.5]],
                    columns=['a', 'b', 'c', 'd', 'e', 'f'])
pd_x_k = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2]],
                      columns=['a', 'b', 'c', 'd', 'e', 'f'])

emm = Confidence(known=pd_x)
pred_variance, confidence = emm.assess_x(pd_x_k)
```
The results are dictionaries with keys matching any responses provided as well
as a 'Full' key which evaluates the row in the setting of all x values
(without regard for missing values in responses). The first element is the
calculated, unscaled prediction variance. The second element is a string of 
'High', 'Mid' or 'Low' indicating how confident you can feel in the model's
ability to make predictions in this space.
* High - the prediction variance is less than the 90th percentile of training
data's prediction variances
* Mid - the prediction variance is no greater than the maximum prediction
variance of the training data
* Low - the prediction variance is greater than the maximum prediction variance
of the training data
