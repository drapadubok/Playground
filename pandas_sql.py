import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("") # link to db

# Making horizontal barplot
# Filter and sort data
highest_rated = comments[comments['avg_score'] > 500].sort(['avg_score'],ascending=False)
highest_rated.plot(x='subr',y='avg_score',kind='barh')
# aggregate using sum and mean, also missing data imputation
comments.pivot_table(index=['subr','author'],values=['avg_score'],columns=['controversiality'],
                     aggfunc=[np.sum,np.mean],
                     dropna=True,fill_value=0, margins=True)
# read sql using pd and sqlalchemy
orders = pd.read_sql_table("Orders", engine)
customers = pd.read_sql_table("Customers", engine)
# join
co = pd.merge(customers, orders, how='inner',left_on='CustomerID',right_on='CustomerID')
# chaining WHERE
orders[(orders['Freight'] >= 55.28) & (orders['Freight'] <= 208.58)]
