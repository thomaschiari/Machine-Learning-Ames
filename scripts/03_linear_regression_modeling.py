#!/usr/bin/env python
# coding: utf-8

# # Building a regression model for predicting house sale prices

# In[1]:


import pickle
import pathlib

import numpy as np
import pandas as pd


# In[2]:


DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)


# In[3]:


clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'


# In[4]:


with open(clean_data_path, 'rb') as file:
    data = pickle.load(file)


# In[5]:


data.info()


# In[6]:


model_data = data.copy()


# ## Preparing the data for the model

# Now that we have the data all cleaned, and all the missing values accounted for, lets focus on transforming the data for the model.
# 
# Lets remember what a model is. 
# 
# - A predictive model is a **set** of functions that receive data as input and produce a prediction, that is, an estimate of the target value as output.
# - **Train** a model is to search the set of candidate functions for one that adequately represents the **training dataset**.
# - The adequacy of a candidate function to the training data is usually determined by a **loss function** that measures how well the predictions of the function match the real values of the target within the training dataset. It is common to define a *loss function per data item* (e.g. absolute error, quadratic error, etc) and to construct the *loss function over the dataset* as the *average prediction loss*.
# 
# Many models are **parametric models**. In this case, each function of the set of functions that makes the model is constructed from a vector of **parameters** that define the function, forming a **parametric function**. For instance: the linear model constructs prediction values out of a linear combination of the input features, plus a constant. The weights of the linear combination plus the constant are the parameters of the model. The set of functions that can be represented by this model is given by all possible values of the vector of parameters that define the function.
# 
# Some models are called **non-parametric models**. These models usually do not have a parametric form (like the linear model). But the terminology is a bit misleading, though: usually these models *do* have parameters, and potentially an open-ended set of them! For instance, consider the "decision tree" model, which is one of the most prominent models of this category. The decision tree may not have a formula for the predicted value, but it does have parameters, many of them: each decision in the tree involves a choice of feature and a threshold level, and those choices must be stored as parameters of the model for use in future predictions.
# 
# Each model has specific requirements for the format of the input data. Most of the time, the minimum requirement is that:
# 
# - All columns are numeric;
# - There are no missing values.
# 
# Some models have extra requirements. For example: the support-vector-machines model requires that the input features have comparable standard deviations - having features that have large discrepancies between features in terms of their order of magnitude (such as a feature in the fractions of unit range and another in the tens of thousands) will result in poor prediction quality.
# 
# And some models may not have any special requirement at all. We will study each of those in detail in this course.
# 
# Lets start our study with a simple model: the *multivariate linear regression* model. This is a model that presents the minimum requirements listed above. So we need to do a bit of processing on the original features:
# 
# - *Numerical features* stay as given;
# - *Categorical features* have to be transformed into numerical features. In order to do so we need to **encode** these features, that is: to transform them into new features that convey the same information, but in a numerical form, and in a way that "makes sense" - we'll see it below.
# - *Ordinal features* can be transformed into numerical features in the same way as the caegorical features, or could be assigned increasing numbers in conformity with the ordered nature of the categories of the feature.

# ## Encoding categorical variables

# Lets identify all categorical variables - both nominal (that is, categoricals without category order) and ordinal.

# In[7]:


categorical_columns = []
ordinal_columns = []
for col in model_data.select_dtypes('category').columns:
    if model_data[col].cat.ordered:
        ordinal_columns.append(col)
    else:
        categorical_columns.append(col)


# In[8]:


ordinal_columns


# In[9]:


categorical_columns


# ### Encoding ordinal variables 

# Ordinal variables can be transformed into integer numbers in a straightforward manner: the lowest category is assigned the value "zero", the next category is given the value "one", etc. The `Pandas` library has a function for this task: `factorize()`:

# In[10]:


for col in ordinal_columns:
    codes, _ = pd.factorize(data[col], sort=True)
    model_data[col] = codes


# Lets confirm that the variables are no longer ordinal, but now are integers:

# In[11]:


model_data[ordinal_columns].info()


# Compare the original values with the encoded values:

# In[12]:


data['Lot.Shape'].value_counts()


# In[13]:


model_data['Lot.Shape'].value_counts()


# ### Encoding nominal variables

# With nominal variables there is no notion of order among categories. Therefore, it would be a conceptual mistake to encode them in the same manner as the ordinal variables. For instance, consider the `Exterior` variable:

# In[14]:


model_data['Exterior'].value_counts()


# We cannot assign an order here, lest we end up with equations like `HdBoard` + `Plywood` = `CemntBd`, which are nonsense. 
# 
# The strategy here to encode `Exterior` is to create several new numerical variables to represent the membership of a given data item to one of the `Exterior` categories. These are called **dummy variables**. Each of these new variables contain only the values "zero" or "one" (i.e. they are binary variables), where $1$ denotes that the data item belongs to the category represented by the variable. Evidently, for a given data item, only one dummy variable has a value of $1$, all remaining are $0$.
# 
# There are two types of dummy variable encoding:
# 
# - "One-hot" encoding: in this case we create one dummy variable per category. Let's look at the `Exterior` feature as an example. The `Pandas` function `get_dummies()` can do the encoding for us:

# In[15]:


original_data = model_data['Exterior']
encoded_data = pd.get_dummies(original_data)

aux_dataframe = encoded_data
aux_dataframe['Exterior'] = original_data.copy()

aux_dataframe.head().transpose()


# Observe that for each value of `Exterior`, only the corresponding dummy is flagged.
# 
# One-hot encoding is a popular technique in Machine Learning. Statisticians, however, prefer a slightly different way of dummy encoding which is:
# 
# - Choose a category to *not encode* (this is called the *base category*)
# - Generate dummies for the remaining categories. That is:
#     - If the data item belongs to the base category, no dummy receives a value of $1$;
#     - Otherwise, set the corresponding dummy to $1$.
# 
# The same `get_dummies()` function of `Pandas` can do this automatically with the `drop_first` argument:

# In[16]:


original_data = model_data['Exterior']
encoded_data = pd.get_dummies(original_data, drop_first=True)

aux_dataframe = encoded_data
aux_dataframe['Exterior'] = original_data.copy()

aux_dataframe.head().transpose()


# Notice that we are now missing the dummy variable for the `AsbShng` category.
# 
# Why to encode things this way? If we don't drop one of the dummies, then it will always be the case that the sum of the values of the dummies is $1$ (since each data item must belong to one of the categories). The linear model, particularly very popular with the statisticians, implies the existence of a fictitious feature containing, for all data items, the value $1$. Hence we end up having a set of variables where a linear combination of them (in this case, the sum of the dummies) matches the value at another variable. This has numerical computing implications for the linear model, that we will discuss in class.
# 
# Since we want to use the linear model in this notebook, lets encode all categoricals with the `drop_first` alternative.

# In[17]:


model_data = pd.get_dummies(model_data, drop_first=True)


# Now our dataset has a lot more variables!

# In[18]:


model_data.info()


# In[19]:


for cat in categorical_columns:
    dummies = []
    for col in model_data.columns:
        if col.startswith(cat + "_"):
            dummies.append(f'"{col}"')
    dummies_str = ', '.join(dummies)
    print(f'From column "{cat}" we made {dummies_str}\n')


# ## Train-test splitting

# The data will now be organized as follows:
# 
# - The features form a matrix $X$ of size $m \times n$, where $m$ is the number of data items, and $n$ is the number of features.
# - The target forms a column-matrix $y$ of length $m$.

# In[20]:


X = model_data.drop(columns=['SalePrice']).copy()
y = model_data['SalePrice'].copy()


# In[21]:


X.values, y.values


# This is the typical set-up of a machine learning project. Now we want to train our model *and* verify that the model provides good predictions for *unseen* data items. Why the emphasis on "unseen"? Because there is no use for a model that only gives predictions for the items in the data used to train it - we want our models to *generalize*.
# 
# The way to assess the model's performance for unseen values is to split the dataset into two subsets: the **training** and **test** datasets.
# 
# We have been using a lot of `Pandas` to manipulate our data so far. From now on we will switch to another very popular library for machine learning in Python: `Scikit-Learn`.
# 
# The function `train_test_split()` will take as arguments the dataset to be split, the specification of the fraction of the dataset to be reserved for testing, and a random seed value - so that the split will always be the same whenever we run our notebook. This is a customary measure to ensure reproducibility of the notebook.

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


RANDOM_SEED = 42  # Any number here, really.


# In[24]:


Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_SEED,
)


# In[25]:


X.shape, Xtrain.shape, Xtest.shape


# In[26]:


y.shape, ytrain.shape, ytest.shape


# ## Fitting a linear model

# Lets start with fitting a linear model for regression. The linear model is one of the oldest and most used models for regression, due to its simplicity and strong statistical roots. A proper statistical approach to the understanding of the linear model consists of:
# 
# - Understanding the statistical premises of the linear model;
# - Analyzing the features to verify that the preliminary conditions of this modeling strategy are satisfied;
# - Fitting the model;
# - Analyzing the residuals to confirm that the post-fit conditions are satisfied.
# 
# Lets discuss these topics in more detail:

# ### The statistical approach to the linear model

# In machine learning we are more interested in the predictive capability of a model, rather than its inductive use to analyze the relation between features and target (or independent and dependent variables, in the statistical terminology). But even to a machine learning practitioner, understanding the statistical basis of the linear model may lead to better predictive performance. For instance:
# 
# - Having a symmetrical residual is usually associated with better mean-squared-error (MSE) than having a long-tailed assymmetric residual;
# - Non-significant parameters (in a hypothesis-testing sense) may indicate superfluous variables in the model, which could be associated with reduced performance in the test dataset (i.e. poor generalization).
# 
# So what is the linear model in statistics? A statistical model is a way to describe probabilistically the relation between features and targets, usually in a parametric way.
# 
# Mathematically, we are searching for a *conditional probability* density model of the form $f(Y = y | \mathbf{x}, \theta)$ where $\mathbf{x}$ is the feature vector, $Y$ is the random variable associated with the target, and $\theta$ is the vector of parameters. In plain language, we would like to describe the probability distribution of the target variable when the value of the feature vector is known and the parameters of the model are given.
# 
# In the linear model, we postulate that the data $y$ is generated from the feature vector $\mathbf{x}$ plus a random Gaussian noise of fixed standard deviation, as shown in the equation below:
# 
# $$
# y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n + \varepsilon
# $$
# 
# where $\varepsilon \sim N(0, \sigma)$
# 
# The addition of the noise term causes the $y$ value to become a random variable itself, thus making the probabilistic model mentioned above. For a given value of $\mathbf{x}$, the value of $y$ is obtained by adding the constant value $\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$ to a normal random variable $\varepsilon$. Remember that, for normal random variables, adding a constant keeps the variable normal, with a shifted mean-value parameter. Therefore:
# 
# $$
# Y \sim N(\mu = (\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n), \sigma)
# $$
# 
# Lets write
# 
# $$
# \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
# $$
# 
# for simplicity, then the model above is rewritten as:
# 
# $$
# Y \sim N(\mu = \hat{y}, \sigma)
# $$
# 
# When we have a dataset $D = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \cdots, (\mathbf{x}_m, y_m)\}$ of several $(\mathbf{x}, y)$ pairs, what is their joint conditional probability density function $f(y_1, y_2, \cdots, y_m | x_1, x_2, \cdots, x_n, \theta)$?
# 
# We will make another supposition of the linear model: that the $(\mathbf{x}, y)$ examples were obtained independently, and that the value of one does not impact the probability of the other. Therefore:
# 
# $$
# f(y_1, y_2, \cdots, y_m | x_1, x_2, \cdots, x_n, \theta) = f(y_1| x_1, \theta) f(y_2| x_2, \theta) \cdots f(y_m| x_m, \theta) 
# $$
# 
# Remember that the normal probability density function is as follows:
# 
# $$
# Y \sim N(\mu, \sigma) \Rightarrow f(y) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{1}{2}\frac{(y - \mu)^2}{\sigma^2} \right)
# $$
# 
# Thus:
# 
# $$
# Y \sim N(\mu = \hat{y}, \sigma) \Rightarrow f(y) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{1}{2}\frac{(y - \hat{y})^2}{\sigma^2} \right)
# $$
# 
# And the joint conditional probability density function of the entire dataset becomes:
# 
# 
# $$
# f(y_1, y_2, \cdots, y_m | x_1, x_2, \cdots, x_n, \theta) = \prod_{i=1}^{m} \left(\frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{1}{2}\frac{(y_i - \hat{y_i})^2}{\sigma^2} \right) \right)
# $$
# 
# Expanding the product we have:
# 
# $$
# \begin{align*}
# f(y_1, y_2, \cdots, y_m | x_1, x_2, \cdots, x_n, \theta) & = & \prod_{i=1}^{m} \left(\frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{1}{2}\frac{(y_i - \hat{y_i})^2}{\sigma^2} \right) \right) \\
# & = & \prod_{i=1}^{m} \left(\frac{1}{\sigma \sqrt{2 \pi}} \right) \prod_{i=1}^{m} \left( \exp \left(-\frac{1}{2}\frac{(y_i - \hat{y_i})^2}{\sigma^2} \right) \right) \\
# & = & \left(\frac{1}{\sigma \sqrt{2 \pi}} \right)^{m} \exp \left(\sum_{i=1}^{m} \left(-\frac{1}{2}\frac{(y_i - \hat{y_i})^2}{\sigma^2} \right) \right) \\
# & = & \left(\frac{1}{\sigma \sqrt{2 \pi}} \right)^{m} \exp \left(- \frac{1}{2 \sigma} \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right)
# \end{align*}
# $$
# 
# What are the "best" value for the parameters of the linear model? We can search for the parameters that maximize the joint conditional probability density function of the dataset. This function is called the *likelihood* of the parameters, and therefore our solution here is called a *"maximum likelihood estimate"* of the parameters.
# 
# So we are looking for a value $\theta^{\star}$ of $\theta$ to maximize $f(y_1, y_2, \cdots, y_m | x_1, x_2, \cdots, x_n, \theta)$, that is:
# 
# $$
# \begin{align*}
# \theta^{\star} & = & \argmax_{\theta} \left\{ f(y_1, y_2, \cdots, y_m | x_1, x_2, \cdots, x_n, \theta) \right\}\\
# & = & \argmax_{\theta} \left\{ \left(\frac{1}{\sigma \sqrt{2 \pi}} \right)^{m} \exp \left(- \frac{1}{2 \sigma} \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right) \right\} \\
# & = & \argmax_{\theta} \left\{ \exp \left(- \frac{1}{2 \sigma} \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right) \right\} \\
# & = & \argmax_{\theta} \left\{ - \frac{1}{2 \sigma} \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right\} \\
# & = & \argmin_{\theta} \left\{ \frac{1}{2 \sigma} \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right\} \\
# & = & \argmin_{\theta} \left\{ \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right\} \\
# & = & \argmin_{\theta} \left\{ \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y_i})^2 \right\} \\
# \end{align*}
# $$
# 
# Hey, look who we found! Our old friend MSE (mean-squared-error)!
# 
# So, in the end, here are the lessons:
# 
# - The statistical formulation of the linear model leads to the same error formulation of machine learning, which only cares for the prediction quality.
# - The statistical linear model has several assumptions:
#     - The samples are independent;
#     - The target is *truly* generated from the linear predictive formula plus a normally-distributed error;
#     - The error has zero mean and constant standard deviation (the *homoscedasticity* hypothesis);
#     - There is no error in the feature measurement. That is, $\mathbf{x}_{i}$ are constants, not random variables. All the error is in the target;
# - If the assumptions of the linear model are satisfied, you can analyze the parameters with greater sophistication (the machine learning formulation does not bring this finesse). For instance, you can run hypothesis tests on the values of the parameters to determine whether they refute the null hypothesis $\theta_i = 0$ with a given statistical significance level. Which, in plain language, means that you don't really trust that the associated feature impacts the target, or if it is just an accident.

# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


model = LinearRegression()

model.fit(Xtrain, ytrain)


# In[29]:


ypred = model.predict(Xtest)


# In[30]:


from sklearn.metrics import mean_squared_error

RMSE = np.sqrt(mean_squared_error(ytest, ypred))


# In[31]:


RMSE


# In[32]:


error_percent = 100 * (10**RMSE - 1)
print(f'Average error is {error_percent:.2f}%')


# In[ ]:




