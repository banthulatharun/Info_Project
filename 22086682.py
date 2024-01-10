#!/usr/bin/env python
# coding: utf-8

# ## Loading the Dataset

# In[18]:


import pandas as pd
import numpy as np


# In[28]:


# Read the CSV file into a DataFrame and specify a column name
df = pd.read_csv("data2-1.csv", header=None, names=['salary'])


# In[29]:


df


# In[30]:


print('The Dataset has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))


# ## Data Information

# In[31]:


df.info(memory_usage='deep')


# ## Statistical Parameters

# In[32]:


df.describe()


# ## Finding Null Values

# In[33]:


df.isnull().sum()


# ## Probability Density Function

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_salary_distribution(df):
    """
    Plots the probability density function of the salary data.

    Parameters:
    - df: DataFrame containing the salary data.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['salary'], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title('Salary Distribution and PDF')
    plt.xlabel('Salary')
    plt.ylabel('Probability Density')
    plt.show()

#  df is already read from the CSV file
plot_salary_distribution(df)


# ## Mean Annual Salary

# In[35]:


def calculate_mean_salary(df):
    """
    Calculates the mean annual salary using the obtained probability density function.

    Parameters:
    - df: DataFrame containing the salary data.

    Returns:
    - mean_salary: Mean annual salary.
    """
    # df is already read from the CSV file
    bin_heights, bin_edges = np.histogram(df['salary'], bins=30, density=True)

    # Calculate the mean salary by summing the product of each salary value and its corresponding probability
    mean_salary = np.sum(bin_heights * (bin_edges[:-1] + bin_edges[1:]) / 2)

    return mean_salary

# Call the function with your DataFrame (df)
mean_salary = calculate_mean_salary(df)
print(f"Mean Annual Salary (˜W): {mean_salary:.2f}")


# ## Calculate Another Value, X

# In[40]:


def calculate_x_for_percentile(df, percentile_value):
    """
    Calculates the salary value (X) below which a certain percentage of people fall.

    Parameters:
    - df: DataFrame containing the salary data.
    - percentile_value: Desired percentile (e.g., 5 for 5%).

    Returns:
    - x_value: Salary value for the specified percentile.
    """
    x_value = np.percentile(df['salary'], 100 - percentile_value)
    return x_value

# Call the functions with DataFrame (df)
mean_salary = calculate_mean_salary(df)
x_value = calculate_x_for_percentile(df, 5)


# In[51]:


# Plotting the histogram
plt.hist(df['salary'], bins=30, density=True, alpha=1.0, color='blue', label='Histogram')

# Print mean_salary and x_value on the graph
plt.text(0.05, 0.95, f'Mean Annual Salary (˜W): {mean_salary:.2f}', transform=plt.gca().transAxes, color='red')
plt.text(0.05, 0.90, f'Salary Value for 5th Percentile (X): {x_value:.2f}', transform=plt.gca().transAxes, color='black')

# Display the graph
plt.xlabel('Salary')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

