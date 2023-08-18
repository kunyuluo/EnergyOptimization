import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

sns.set()

# data = pd.read_csv("C:\\Users\\DELL\\Downloads\\real_estate_price_size_year.csv")
data = pd.read_csv("D:\\ladybug\\Office_Parametric\\data.csv")

y = data['KWH'] / 1000
x1 = data[['Wall-U', 'SHGC', 'LPD', 'Clg-COP']]

# x = sm.add_constant(x1)
# results = sm.OLS(y, x).fit()
# print(results.summary())

list = ['lakers', 'celtics', 'suns', 'denver']
chance = [2, 1, 1, 1]
team = random.choices(list, weights=chance)
print(team)
