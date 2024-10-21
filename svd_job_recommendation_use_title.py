#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install scikit-surprise


# In[2]:


import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[3]:


df = pd.read_csv('linkdin_job_posting.csv')

# Display the first few rows to understand the structu
df.head()


# In[4]:


df.info()


# In[5]:


# Selecting relevant columns for the recommendation task
relevant_columns = ['title', 'views', 'applies']

# Dropping rows with missing values in the relevant columns
cleaned_data = df[relevant_columns].dropna()

# Display the cleaned dataset to confirm
cleaned_data.head()


# In[6]:


cleaned_data.shape


# In[7]:


# For simplicity, we'll simulate 1000 unique users
np.random.seed(42)  # For reproducibility
num_users = 1000
user_ids = np.random.randint(1, num_users + 1, size=len(cleaned_data))


# In[8]:


# Add the simulated User_ID to the dataset
cleaned_data['User_ID'] = user_ids


# In[9]:


#Fill missing values in 'applies' or 'views' with 0 (representing no interaction)
cleaned_data['applies'].fillna(0, inplace=True)
cleaned_data['views'].fillna(0, inplace=True)


# In[10]:


#Decide which interaction to use (we'll use 'applies' for this example)
cleaned_data['interaction'] = cleaned_data['applies']


# In[11]:


# Drop rows where interaction data is missing or 0 (optional step based on model needs)
cleaned_data = cleaned_data[cleaned_data['interaction'] > 0]
cleaned_data.head()


# In[12]:


# Final dataset ready for SVD training
cleaned_data = cleaned_data[['User_ID', 'title', 'interaction']]

cleaned_data.head()


# In[13]:


# Save the cleaned dataset (optional, if you want to save the preprocessed data)
cleaned_data.to_csv('cleaned_job_posting_for_svd_use_title.csv', index=False)


# **Model Training**

# In[14]:


# Check the range of 'applies' to determine the correct rating scale
print(cleaned_data['interaction'].min(), cleaned_data['interaction'].max())


# In[15]:


# Step 1: Prepare the data for SVD in Surprise format
reader = Reader(rating_scale=(1, 967))  # Assuming the rating scale ranges from 1 to 10
data = Dataset.load_from_df(cleaned_data[['User_ID', 'title', 'interaction']], reader)


# In[16]:


# Step 2: Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)


# In[17]:


# Step 3: Train the SVD model
model = SVD()
model.fit(trainset)


# In[18]:


# Step 4: Evaluate the model on the test set using RMSE (optional)
predictions = model.test(testset)
accuracy.rmse(predictions)


# In[19]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Extract the true ratings and predicted ratings
true_ratings = [pred.r_ui for pred in predictions]  # Actual ratings
predicted_ratings = [pred.est for pred in predictions]  # Predicted ratings

# Mean Absolute Error (MAE)
mae = mean_absolute_error(true_ratings, predicted_ratings)

# Mean Squared Error (MSE)
mse = mean_squared_error(true_ratings, predicted_ratings)

# R-squared (R²)
r2 = r2_score(true_ratings, predicted_ratings)

# RMSE
rmse = accuracy.rmse(predictions)

# Print the evaluation metrics
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")


# I will run hyperparameter optimization

# In[20]:


#pip install optuna


# In[21]:


import optuna
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import pandas as pd

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 967))  # Assuming the rating scale ranges from 1 to 10
data = Dataset.load_from_df(cleaned_data[['User_ID', 'title', 'interaction']], reader)

# Split the data into trainset and testset (this will be done within the cross-validation)
trainset = data.build_full_trainset()

# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to be tuned
    param = {
        'n_factors': trial.suggest_int('n_factors', 10, 200),  # Number of latent factors
        'n_epochs': trial.suggest_int('n_epochs', 5, 50),  # Number of epochs
        'lr_all': trial.suggest_float('lr_all', 0.001, 0.05),  # Learning rate for SGD
        'reg_all': trial.suggest_float('reg_all', 0.001, 0.1)  # Regularization term
    }

    # Create the SVD model with the suggested hyperparameters
    svd_model = SVD(**param)

    # Perform cross-validation and return the RMSE score
    results = cross_validate(svd_model, data, measures=['rmse'], cv=5, verbose=False)
    
    # Return the average RMSE score (minimizing this)
    return np.mean(results['test_rmse'])

# Enable the default logger of Optuna
optuna.logging.enable_default_handler()

# Set the logging level (for more detailed information)
optuna.logging.set_verbosity(optuna.logging.INFO)

# Create an Optuna study for minimizing the RMSE (Root Mean Squared Error)
study = optuna.create_study(direction='minimize')

# Perform optimization
study.optimize(objective, n_trials=100)

# Best hyperparameters
print('Best trial:', study.best_trial.params)

# Best RMSE value
print('Best RMSE:', study.best_value)



# In[22]:


best_params = study.best_trial.params
print("Best hyperparameters: ", best_params)


# In[23]:


# retrain the model with the tuned hyperparameters

# Step 3: Train the SVD model
model_tune = SVD(**best_params)
model_tune.fit(trainset)


# In[24]:


# Step 4: Evaluate the model on the test set using RMSE (optional)
predictions = model_tune.test(testset)
accuracy.rmse(predictions)


# In[25]:


# Extract the true ratings and predicted ratings
true_ratings = [pred.r_ui for pred in predictions]  # Actual ratings
predicted_ratings = [pred.est for pred in predictions]  # Predicted ratings

# Mean Absolute Error (MAE)
mae = mean_absolute_error(true_ratings, predicted_ratings)

# Mean Squared Error (MSE)
mse = mean_squared_error(true_ratings, predicted_ratings)

# R-squared (R²)
r2 = r2_score(true_ratings, predicted_ratings)

# RMSE
rmse = accuracy.rmse(predictions)

# Print the evaluation metrics
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")


# The hyperparameter tuning improved the model

# In[26]:


# Step 5: Save the trained model using joblib for later use (e.g., deployment)

import joblib
model_filename = 'svd_model_job_use_title.joblib'
joblib.dump(model_tune, model_filename)
print(f"Model saved as {model_filename}")


# In[ ]:




