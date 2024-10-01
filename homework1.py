import pandas as pd
import numpy as np
import os

# Q1. Pandas version
print("Q1. What's the version of Pandas that you installed?")
print(f"A1. Pandas version: {pd.__version__}")

# Q2. Records count
print("\nQ2. How many records are in the dataset?")

data = "/mnt/c/Users/USER/Downloads/laptops.csv"

df = pd.read_csv(data)

num_of_rows = df.shape[0]

print(f"A2. The total records of the dataset is: {num_of_rows}")

# Q3. Laptop brands
print("\nQ3. How many laptop brands are presented in the dataset?")

count_brand = df['Brand'].nunique()

print(f"A3. The total of laptop brands that presented in the dataset are: {count_brand}")

# Q4. Missing values
print("\nQ4. How many columns in the dataset have missing values?")

missing_values = df.isnull().sum()

print(f"A4. The list of columns in the dataset that have missing values are: \n{missing_values}")

total_cols_missing = len([val for val in missing_values if val!=0])

print(f"\nThe total of columns that have missing values are: {total_cols_missing}")

# Q5. Maximum final price
print("\nQ5. What's the maximum final price of Dell notebooks in the dataset?")

max_price = df[df['Brand']=='Dell']['Final Price'].max()
print(f"A5. The maximum final price of Dell notebooks is: {max_price}")

# Q6. Median value of Screen
print("\nQ6. Find the median value of Screen column and calculate the median value of Screen. Has it changed?")

# 1. The median value of Screen column
med_screen = df['Screen'].median()
print(f"1st median value: {med_screen}")

# 2. Calculate the most frequent value of the same Screen column
mode_screen = df['Screen'].mode()
print(f"Most frequent value of the same Screen: {mode_screen}")

# 3. Use fillna method to fill the missing values in Screen column with the most frequent value from the previous step
fill_missing = df.fillna(mode_screen)

# 4. Now, calculate the median value of Screen once again
med_screen2 = df['Screen'].median()
print(f"2nd median value: {med_screen2}")

if med_screen == med_screen2:
    print("\nThere's no change between the 1st and the 2nd median.")
else:
    print("\nThe 2nd median has changed.")

# Q7. Sum of weights
print("\nQ7. What's the sum of all the elements of the result?")

# 1. Select all the "Innjoo" laptops from the dataset
injoo_laptops = df[df['Brand']=='Innjoo']

# 2. Select only columns RAM, Storage, Screen
injoo_laptops_sel_cols = injoo_laptops[['RAM', 'Storage', 'Screen']]
print(f"A7. \nList Injoo laptops selected columns: \n{injoo_laptops_sel_cols}")

# 3. Get the underlying NumPy array. Let's call it X
X = injoo_laptops_sel_cols.to_numpy()
print(f"\nUnderlying NumPy array: \n{X}")

# 4. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX
XTX = X.T @ X
print(f"\nMultiplication between the transpose of X and X: \n{XTX}")

# 5. Compute the inverse of XTX
inv_XTX = np.linalg.inv(XTX)
print(f"\nInverse of XTX: \n{inv_XTX}")

# 6. Create an array y with values [1100, 1300, 800, 900, 1000, 1100]
y = [1100, 1300, 800, 900, 1000, 1100]
print(f"\nArray y: \n{y}")

# 7. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w
w = (inv_XTX @ X.T) * y
print(f"\nMultiplication of the inverse of XTX & the transpose of X, and multiply with array y: \n{w}")

# 8. What's the sum of all the elements of the result?
sum = np.sum(w)
print(f"\nThe sum of all the elements of the result is: {round(sum, 3)}")