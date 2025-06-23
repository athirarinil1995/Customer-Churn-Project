#!/usr/bin/env python
# coding: utf-8

# # Project : Customer Churn Dataset

# ## Problem Statement:

#  You are the data scientist at a telecom company named “Neo” whose customers
#  are churning out to its competitors. You have to analyze the data of your
#  company and find insights and stop your customers from churning out to other
#  telecom companies.

# ## Customer_churn Dataset:

#  The details regarding this ‘customer_churn’ dataset are present in the data dictionary

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("C:\\Users\\athir\\Downloads\\customer_churn.csv")


# In[4]:


df.head()


# In[ ]:





# # Lab Environment: Anaconda

# ## Domain: Telecom

# ## Tasks To Be Performed:

# ### 1. Data Manipulation:

#  1.Extract the 5th column and store it in ‘customer_5’

# In[6]:


column_names = df.columns
print(column_names)


# In[13]:


customer_5 = df.iloc[:,4]


# In[14]:


customer_5


# 2.Extract the 15th column and store it in ‘customer_15’

# In[15]:


customer_15 = df.iloc[:,14]


# In[16]:


customer_15


# 3.Extract all the male senior citizens whose payment method is electronic check and store the result in ‘senior_male_electronic’

# In[23]:


senior_male_electronic = df[(df['gender'] == 'Male') & (df['SeniorCitizen'] == 1) & (df['PaymentMethod'] == 'Electronic check')]


# In[24]:


senior_male_electronic


# 4.Extract all those customers whose tenure is greater than 70 months or their monthly charges is more than $100 and store the result in ‘customer_total_tenure’

# In[36]:


customer_total_tenure = df[(df['tenure'] > 70) | (df['MonthlyCharges'] > 100) ]


# In[37]:


customer_total_tenure


#  5. Extract all the customers whose contract is of two years, payment method
#  is mailed check and the value of churn is ‘Yes’ and store the result in
#  'two_mail_yes'

# In[39]:


two_mail_yes = df[(df['Contract'] == 'Two year') & (df['PaymentMethod'] == 'Mailed check') &  (df['Churn'] == 'Yes')]


# In[40]:


two_mail_yes


# 6.Extract 333 random records from the customer_churndataframe and store
#  the result in ‘customer_333’

# In[42]:


customer_333 = df.sample(n=333)
print(customer_333)


# 7.Get the count of different levels from the ‘Churn’ column

# In[43]:


churn_counts = df['Churn'].value_counts()

print(churn_counts)


# ### 2. Data Visualization:

# 1. Build a bar-plot for the ’InternetService’ column:
#  a. Set x-axis label to ‘Categories of Internet Service’
#  b. Set y-axis label to ‘Count of Categories’
#  c. Set the title of plot to be ‘Distribution of Internet Service’
#  d. Set the color of the bars to be ‘orange’

# In[44]:


internet_service_counts = df['InternetService'].value_counts()


# In[45]:


internet_service_counts


# In[46]:


internet_service_counts.plot(kind='bar', color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()


# Build a histogram for the ‘tenure’ column:
#  a. Set the number of bins to be 30
#  b. Set the color of the bins to be ‘green’
#  c. Assign the title ‘Distribution of tenure’

# In[47]:


df['tenure'].hist(bins=30, color='green')

# Assign the title to the plot
plt.title('Distribution of tenure')

# Display the plot
plt.show()


# 3.Build a scatter-plot between ‘MonthlyCharges’ and ‘tenure’. Map
#  ‘MonthlyCharges’ to the y-axis and ‘tenure’ to the ‘x-axis’:
#  a. Assign the points a color of ‘brown’
#  b. Set the x-axis label to ‘Tenure of customer’
#  c. Set the y-axis label to ‘Monthly Charges of customer’
#  d. Set the title to ‘Tenure vs Monthly Charges’
#  e. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the
#  y-axis &
#  f. ‘Contract’ on the x-axis.

# In[50]:


# Create the scatter plot between 'MonthlyCharges' and 'tenure'
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.scatter(df['tenure'], df['MonthlyCharges'], color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()

# Create the box plot between 'tenure' and 'Contract'
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.boxplot(x='Contract', y='tenure', data=df)
plt.xlabel('Contract')
plt.ylabel('Tenure')
plt.title('Box Plot between Tenure and Contract')
plt.show()


# ### 3. Linear Regression:
 Build a simple linear model where dependent variable is ‘MonthlyCharges’
 and independent variable is ‘tenure’:
 a. Divide the dataset into train and test sets in 70:30 ratio.
 b. Build the model on train set and predict the values on test set
 c. After predicting the values, find the root mean square error
 d. Find out the error in prediction & store the result in ‘error’
 e. Find the root mean square error
# In[51]:


df.info()


# In[52]:


df.isnull().sum()


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# In[54]:


# Define the independent and dependent variables
X = df[['tenure']]
y = df['MonthlyCharges']


# In[55]:


# Split the dataset into train and test sets in a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[56]:


# Standardize the independent variable (tenure)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[57]:


# Build the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# In[58]:


# Predict the values on the test set
y_pred = model.predict(X_test_scaled)


# In[59]:


# Calculate the root mean square error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)


# In[60]:


# Calculate the error in prediction and store the result in 'error'
error = y_test - y_pred


# In[61]:


# Display the RMSE and error
print(f"Root Mean Square Error (RMSE): {rmse}")
print("Error in prediction:", error)


# ### 4. Logistic Regression:
Build a simple logistic regression model where dependent variable is
 ‘Churn’ and independent variable is ‘MonthlyCharges’:
 a. Divide the dataset in 65:35 ratio
 b. Build the model on train set and predict the values on test set
 c. Build the confusion matrix and get the accuracy score
 d. Build a multiple logistic regression model where dependent variable
 is
 ‘Churn’ and independent variables are ‘tenure’ and
 ‘MonthlyCharges’
 e. Divide the dataset in 80:20 ratio
 f. Build the model on train set and predict the values on test set
 g. Build the confusion matrix and get the accuracy score
# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[69]:


# Simple Logistic Regression (MonthlyCharges as independent variable)
# Split the dataset in 65:35 ratio
X_simple = df[['MonthlyCharges']]
y = df['Churn']
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y, test_size=0.35, random_state=1)


# In[70]:


# Build the logistic regression model
model_simple = LogisticRegression()
model_simple.fit(X_train_simple, y_train_simple)


# In[71]:


# Predict values on the test set
y_pred_simple = model_simple.predict(X_test_simple)


# In[72]:


# Calculate the confusion matrix and accuracy score
confusion_matrix_simple = confusion_matrix(y_test_simple, y_pred_simple)
accuracy_simple = accuracy_score(y_test_simple, y_pred_simple)

print("Simple Logistic Regression Model")
print("Confusion Matrix:")
print(confusion_matrix_simple)
print(f"Accuracy: {accuracy_simple}")


# In[81]:


# Multiple Logistic Regression (tenure and MonthlyCharges as independent variables)
# Split the dataset in 80:20 ratio
X_multiple = df[['tenure', 'MonthlyCharges']]
y = df['Churn']
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y, test_size=0.20, random_state=1)


# In[82]:


# Build the logistic regression model
model_multiple = LogisticRegression()
model_multiple.fit(X_train_multiple, y_train_multiple)


# In[83]:


# Predict values on the test set
y_pred_multiple = model_multiple.predict(X_test_multiple)


# In[84]:


# Calculate the confusion matrix and accuracy score
confusion_matrix_multiple = confusion_matrix(y_test_multiple, y_pred_multiple)
accuracy_multiple = accuracy_score(y_test_multiple, y_pred_multiple)

print("\nMultiple Logistic Regression Model")
print("Confusion Matrix:")
print(confusion_matrix_multiple)
print(f"Accuracy: {accuracy_multiple}")


# ### 5. Decision Tree:
 ● Build a decision tree model where dependent variable is ‘Churn’ and
 independent variable is ‘tenure’:
 a. Divide the dataset in 80:20 ratio
 b. Build the model on train set and predict the values on test set
 c. Build the confusion matrix and calculate the accuracy
# In[91]:


from sklearn.tree import DecisionTreeClassifier
# Define the independent and dependent variables
X = df[['tenure']]
y = df['Churn']

# Split the dataset into train and test sets in an 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[92]:


# Build the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[93]:


# Predict values on the test set
y_pred = model.predict(X_test)


# In[94]:


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")


# ### 6. Random Forest:
 ● Build a Random Forest model where dependent variable is ‘Churn’ and
 independent variables are ‘tenure’ and ‘MonthlyCharges’:
 a. Divide the dataset in 70:30 ratio
 b. Build the model on train set and predict the values on test set
 c. Build the confusion matrix and calculate the accuracy
# In[105]:


from sklearn.ensemble import RandomForestClassifier
# Define the independent and dependent variables
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']


# In[106]:


# Split the dataset into train and test sets in a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[107]:


# Build the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[108]:


# Predict values on the test set
y_pred = model.predict(X_test)


# In[109]:


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:





# In[ ]:




