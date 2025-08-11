import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn .preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer
from sklearn.compose import ColumnTransformer




df = pd.read_csv('housing.csv',encoding='latin1')


decription = df.describe()

missing_values = df.isnull().sum()

df.drop('total_bedrooms',inplace=True,axis=1)



x = df.drop('median_house_value',axis=1)
y = df['median_house_value']

numerical = x.select_dtypes(include=np.number).columns
categorial = x.select_dtypes(include='object').columns

correlations = df[numerical].corr()

preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorial)
    ]
)


x_preprocessed = preprocessor.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_preprocessed,y,random_state=42,test_size=0.33)


#Model Comparison


rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

results = []

#Linear Regresson
lg = LinearRegression()
lg.fit(x_train,y_train)
y_pred_lg = lg.predict(x_test)
rmse_lg = np.sqrt(mean_squared_error(y_test,y_pred_lg))
mae_lg = mean_absolute_error(y_test,y_pred_lg)


print("RMSE Linear Regression : -", rmse_lg)
print("MAE Linear Regression : -", mae_lg)
results.append({
    'Model' : 'Linear Regression',
    'RMSE' : rmse_lg,
    'MAE' : mae_lg
})

print("-" * 30)

#Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)

dt_param_grid = {
    'max_depth' : [10,20,30,None],
    'min_samples_split' : [2,10,20],
    'min_samples_leaf' : [1,5,10]
}

dt_grid_search = GridSearchCV(
    estimator=dt,
    param_grid=dt_param_grid,
    scoring={'RMSE' : rmse_scorer, 'MAE' : mae_scorer},
    refit='RMSE',
    cv=5,
    verbose=1,
    n_jobs = -1
)

dt_grid_search.fit(x_train,y_train)
best_dt_est = dt_grid_search.best_estimator_

y_pred_dt = best_dt_est.predict(x_test)

rmse_dt = np.sqrt(mean_squared_error(y_test,y_pred_dt))
mae_dt = mean_absolute_error(y_test,y_pred_dt)

print("RMSE Decision Tree : -", rmse_dt)
print('MAE Decision Tree:-',mae_dt)
results.append({'Model' : 'Decision Tree', 'RMSE' : rmse_dt, 'MAE' : mae_dt, 'Best Params' : dt_grid_search.best_params_})
print("-" * 30)

#Random Forest Regressor

rr = RandomForestRegressor(random_state=42)


rr_param_grid = {
    'n_estimators' : [100,200,300],
    'max_depth' : [10,20,None],
    'min_samples_split' : [2,5],
    'min_samples_leaf' : [1,2]
}

rr_grid_search = GridSearchCV(
    estimator=rr,
    param_grid=rr_param_grid,
    scoring = {'RMSE' : rmse_scorer , 'MAE' : mae_scorer},
    refit = 'RMSE',
    cv = 5,
    verbose = 1,
    n_jobs = -1
)

rr_grid_search.fit(x_train,y_train)
best_rr_est = rr_grid_search.best_estimator_

y_pred_rr = best_rr_est.predict(x_test)
rmse_rr = np.sqrt(mean_squared_error(y_test,y_pred_rr))
mae_rr = mean_absolute_error(y_test,y_pred_rr)

print('RMSE OF RR:-',rmse_rr)
print('MAE of rr:-',mae_rr)
results.append({'Model' : 'Random Forest Rergression', 'RMSE' : rmse_rr, 'MAE' : mae_rr , 'Best Params' : rr_grid_search.best_params_})


print("--" * 15)


comparison_df = pd.DataFrame(results)






print('\n Summary of model performacnce')
print(comparison_df)

#EDA Print Summaries
print("Missing Values:-",missing_values)
print("Decsription:-",decription)
print("\n Final:-",df.head())

#Visuals
plt.figure()
sns.heatmap(correlations)
plt.title('Correlations')


plt.figure()
sns.scatterplot(data=df,x='median_income',y='median_house_value')
plt.title('Median Income VS Median House Value')

plt.show()

