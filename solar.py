import matplotlib
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')


DATA_URL =('solarpowergeneration.csv')
def load_data(nrows):
    df=pd.read_csv(DATA_URL, nrows=nrows)
    lowercase =lambda x : str(x).lower()
    df.rename(lowercase, axis ='columns', inplace=True)
    return df

data_load_state = st.text('Loading data...')
df = load_data(10000)
st.cache_data()
data_load_state.text('Done !')

st.subheader("2 Fold Ensemble For Solar Power Prediction")
st.markdown('##')
st.markdown('##')

st.success("Dataset used for training and prediction")
st.write(df)
st.markdown('##')
st.markdown('##')


matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

st.success(' Temperature (Celcius) Vs Power Generated (Jules)')
fig=plt.figure()
plt.plot(df['temperature'], df['power-generated'])
st.pyplot(fig)


st.markdown('##')
st.markdown('##')



st.success("Humidity (%) Vs Power generated (Jules)")
fig=plt.figure()
plt.plot(df['humidity'], df['power-generated'],'o')
st.pyplot(fig)

st.markdown('##')
st.markdown('##')


st.success("Wind speed (m/s) Vs Power generated (Jules)")
fig=plt.figure()
plt.plot(df['wind-speed'], df['power-generated'], 'o')
st.pyplot(fig)

st.markdown('##')
st.markdown('##')


st.success("Sky cover Vs Power generated (Jules)")
fig =plt.figure()
plt.plot(df['sky-cover'], df['power-generated'], 'o')
st.pyplot(fig)

st.markdown('##')
st.markdown('##')


st.success("Correlation matrix")

corr = df.corr()
#corr.style.background_gradient(cmap='Greens').set_precision(2)
fig=plt.figure()
plt.figure(figsize = (8, 8))
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2f', cbar=False)
#h.set_yticklabels(h.get_yticklabels(), rotation = 0)
#h.xaxis.tick_top()
#h.figure.savefig(YOURPATH, bbox_inches='tight')
st.pyplot(fig)


st.markdown('##')
st.markdown('##')


st.info("Actual value and the predicted value")


# random regressor doesn't take NaNs so replace them
df.isnull().sum()
df.fillna(0, inplace=True)

#split x and y
x = df.drop('power-generated', axis=1)
y = df['power-generated']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Evaluating the Algorithm

from sklearn import metrics
st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

st.markdown('##')
st.markdown('##')

regressor = RandomForestRegressor(n_estimators = 40, random_state = 0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


st.markdown('##')
st.markdown('##')

regressor = RandomForestRegressor(n_estimators = 26, random_state = 0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



df['date'] = pd.date_range(start='1/1/2019', periods=len(df), freq='8H')


y_test.sort_index().rolling(40).mean().shape, df.rolling(40).mean().shape


st.markdown('##')
st.markdown('##')

#Actual value and the predicted value
df = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
df.head()
st.write(df)



st.markdown('##')
st.markdown('##')


st.success("Scatter Plot Showing The Actual Vs Predicted Power Generated")
st.markdown('##')

fig=plt.figure()
plt.plot(df['Predicted value'].rolling(40).mean(), df['Actual value'].rolling(40).mean(), 'o')
plt.title('Actual vs Predicted Power Generated')
plt.xlabel('Predicted power output (Jules)')
plt.ylabel('Actual power output (Jules)')
st.pyplot(fig)

st.markdown('##')
st.markdown('##')


fig=plt.figure()
s = pd.Series(y_pred)
plt.plot(s.rolling(40).mean()) 
st.pyplot(fig)




