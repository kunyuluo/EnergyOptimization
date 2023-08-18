import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv("C:\\Users\\DELL\\Downloads\\real_estate_price_size_year.csv")

y = data['price']
x1 = data[['size', 'year']]

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())
