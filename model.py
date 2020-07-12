#!/usr/bin/env python
# coding: utf-8

# <img src="https://rhyme.com/assets/img/logo-dark.png" align=center> <h2 align=center> Machine Learning Visualization Tools </h2>

#  

# ### About the Dataset:

# *Concrete Compressive Strength Dataset*
# 
# Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. 
# - Number of instances 1030
# - Number of Attributes 9
# - Attribute breakdown 8 quantitative input variables, and 1 quantitative output variable 

# The aim of the dataset is to predict concrete compressive strength of high performance concrete (HPC). HPC does not always means high strength but covers all kinds of concrete for special applications that are not possible with standard concretes. Therefore, our target value is:

# **Target y**
# - Concrete compressive strength [MPa]
# 
# In this case the compressive strength is the cylindrical compressive strength meaning a cylindrical sample (15 cm diameter; 30 cm height) was used for testing. The value is a bit smaller than testing on cubic samples. Both tests assess the uniaxial compressive strength. Usually, we get both values if we buy concrete.
# 
# To predict compressive strengths, we have these features available:

# **Input X**:
# - Cement $[\frac{kg}{m^3}]$
# - Blast furnace slag $[\frac{kg}{m^3}]$
# - Fly ask $[\frac{kg}{m^3}]$
# - Water $[\frac{kg}{m^3}]$
# - Plasticizer $[\frac{kg}{m^3}]$
# - Coarse aggregate $[\frac{kg}{m^3}]$
# - Fine aggregate $[\frac{kg}{m^3}]$
# - Age $[d]$

# ### Task 1: Introduction 

# In[19]:


# Standard imports
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import warnings
import numpy as np
from pylab import rcParams
import seaborn as sns; sns.set(style="ticks", color_codes=True)
rcParams['figure.figsize'] = 15, 10
import pickle 

warnings.simplefilter('ignore')


#  

# ### Task 2: Dataset Exploration

# In[20]:


# Load the data

df = pd.read_csv('concrete.csv')
your_list = [(600, 0.1, 0.0, 162.0, 2.6, 1001.45, 765, 2)]
my_df = pd.DataFrame(your_list,columns=["cement","slag", "ash", "water", "splast", "coarse", "fine", "age"])
my_df.head()


# In[21]:


df.describe()


#  

# ### Task 3: Preprocessing the Data

# In[22]:


# Specify the features and target of interest
features = [
    "cement","slag","ash","water","splast","coarse","fine","age"
]
target = 'strength'

# Get the X and y data from the DataFrame
X = df[features]
y = df[target]


#  

#  

# ### Task 4: Pairwise Scatterplot

# In[23]:


sns.pairplot(df);


#  

# ### Task 5: Feature Importances

# In[24]:


from yellowbrick.features import FeatureImportances
from sklearn.linear_model import Lasso

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot()

# Title case the feature for better display and create the visualizer
labels = list(map(lambda s: s.title(), features))
viz = FeatureImportances(Lasso(), ax=ax, labels=labels, relative=False)

# Fit and show the feature importances
viz.fit(X, y)
viz.poof()


# ### Task 6: Target Visualization

# In[25]:


from yellowbrick.target import BalancedBinningReference

# Instantiate the visualizer
visualizer = BalancedBinningReference()

visualizer.fit(y)          # Fit the data to the visualizer
visualizer.poof()          # Draw/show/poof the data


# ### Task 7: Evaluating Lasso Regression

# In[26]:


from yellowbrick.regressor import PredictionError
from sklearn.model_selection import train_test_split


# In[27]:


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test


# In[31]:


import joblib

visualizer = PredictionError(Lasso(), size=(800, 600))
visualizer.fit(X_train, y_train)

#regression_model = pickle.dumps(visualizer) 
joblib.dump(visualizer, "regression_model")




#knn_from_pickle = pickle.loads(regression_model) 

#knn_from_pickle.score(X_test, y_test)
#prediction = knn_from_pickle.predict(my_df)

# Call finalize to draw the final yellowbrick-specific elements
visualizer.finalize()

print(prediction)

# Get access to the axes object and modify labels
visualizer.ax.set_xlabel("measured concrete strength")
visualizer.ax.set_ylabel("predicted concrete strength");


# ### Task 8: Visualization of Test-set Errors
# 
# Using YellowBrick we can show the residuals (difference between the predicted value and the truth) both for the training set and the testing set (respectively blue and green).
# 

# In[39]:


from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(Lasso(), size=(800,600))

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

g = visualizer.poof()     # Draw/show/poof the data


# ### Task 9: Cross Validation Scores

# In[36]:


from sklearn.model_selection import KFold
from yellowbrick.model_selection import CVScores

# Create a new figure and axes
_, ax = plt.subplots()

cv = KFold(12)

oz = CVScores(
    Lasso(), ax=ax, cv=cv, scoring='r2'
)

x = oz.fit(X_train, y_train)

oz.poof()


# ### Task 10: Learning Curves

# In[38]:


from yellowbrick.model_selection import LearningCurve
from sklearn.linear_model import LassoCV
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

# Create the learning curve visualizer
sizes = np.linspace(0.3, 1.0, 10)

# Create the learning curve visualizer, fit and poof
viz = LearningCurve(LassoCV(), train_sizes=sizes, scoring='r2')
viz.fit(X, y)

viz.poof()


#  

# ### Task 11:  Hyperparamter Tuning

# The `AlphaSelection` Visualizer demonstrates how different values of alpha influence model selection during the regularization of linear models.

# In[16]:


from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection

# Create a list of alphas to cross-validate against
alphas = np.logspace(-10, 1, 400)

# Instantiate the linear model and visualizer
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model, size=(800,600))

visualizer.fit(X, y)
g = visualizer.poof()


# In[ ]:




