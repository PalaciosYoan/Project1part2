import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.indexes import numeric
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from rgf.sklearn import RGFRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


airbnb_london_listing = './londonairbnb/listings detailed.csv'
airbnb_data = pd.read_csv(airbnb_london_listing)
airbnb_data['price'] = airbnb_data['price'].str.replace('$', '', regex=True)
airbnb_data['price'] = airbnb_data['price'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['weekly_price'] = airbnb_data['weekly_price'].str.replace(
    '$', '', regex=True)
airbnb_data['weekly_price'] = airbnb_data['weekly_price'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['monthly_price'] = airbnb_data['monthly_price'].str.replace(
    '$', '', regex=True)
airbnb_data['monthly_price'] = airbnb_data['monthly_price'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['security_deposit'] = airbnb_data['security_deposit'].str.replace(
    '$', '', regex=True)
airbnb_data['security_deposit'] = airbnb_data['security_deposit'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['cleaning_fee'] = airbnb_data['cleaning_fee'].str.replace(
    '$', '', regex=True)
airbnb_data['cleaning_fee'] = airbnb_data['cleaning_fee'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['extra_people'] = airbnb_data['extra_people'].str.replace(
    '$', '', regex=True)
airbnb_data['extra_people'] = airbnb_data['extra_people'].str.replace(
    ',', '', regex=True).astype(float)

columntf = ['require_guest_profile_picture', 'require_guest_phone_verification', 'host_is_superhost', 'host_has_profile_pic',
            'host_identity_verified', 'is_location_exact', 'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready']

airbnb_data[columntf] = airbnb_data[columntf].replace({'t': 1, 'f': 0})


#print(airbnb_data.columns[airbnb_data.isnull().mean() > 0.50])

missing_more_than_50 = list(
    airbnb_data.columns[airbnb_data.isnull().mean() > 0.50])

airbnb_data = airbnb_data.drop(missing_more_than_50, axis=1)

numerical_ix = airbnb_data.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean'))
])

airbnb_data[numerical_ix] = numeric_transformer.fit_transform(
    airbnb_data[numerical_ix])

categorical_ix = airbnb_data.select_dtypes(include=['object']).columns
# airbnb_data = airbnb_data.drop(categorical_ix, axis=1)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='most_frequent'))
])

airbnb_data[categorical_ix] = categorical_transformer.fit_transform(
    airbnb_data[categorical_ix])


airbnb_data['host_acceptance_rate'] = airbnb_data['host_acceptance_rate'].str.replace(
    '%', '', regex=True).astype(float)
airbnb_data['host_acceptance_rate'] = airbnb_data['host_acceptance_rate'] * 0.01

ids = ['id', 'scrape_id', 'host_id']
airbnb_data = airbnb_data.drop(ids, axis=1)

urls = ['listing_url', 'picture_url',
        'host_url', 'host_thumbnail_url', 'host_picture_url']

airbnb_data = airbnb_data.drop(urls, axis=1)

# summary and description have same info, but description has less null values
# same with street and city
columnsimiliarinfo = ['summary', 'street', 'neighbourhood']
airbnb_data = airbnb_data.drop(columnsimiliarinfo, axis=1)

columnunecessary = ['market', 'country_code',
                    'country', 'smart_location', 'name', 'state', 'city']
airbnb_data = airbnb_data.drop(columnunecessary, axis=1)

columndates = ['last_scraped', 'calendar_last_scraped',
               'first_review', 'last_review', 'host_since', 'calendar_updated']
airbnb_data = airbnb_data.drop(columndates, axis=1)

airbnb_price = airbnb_data['price']
airbnb_data = airbnb_data.drop(columns=['price'], axis=1)

# print(airbnb_data.info())
# fig = plt.figure(figsize=(15, 15))
# ax = fig.gca()
# airbnb_data.hist(ax=ax)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    airbnb_data, airbnb_price, test_size=0.1, random_state=42)

print("Starting rgf")
rgf = RGFRegressor(max_leaf=100,
                   algorithm="RGF_Sib",
                   test_interval=100,
                   loss="LS",
                   verbose=False)


rf = RandomForestRegressor(n_estimators=100,
                           random_state=42)
n_folds = 3
rgf_scores = cross_val_score(rgf,
                             X_train,
                             y_train,
                             scoring=make_scorer(mean_squared_error),
                             cv=n_folds)
rf_scores = cross_val_score(rf,
                            X_train,
                            y_train,
                            scoring=make_scorer(mean_squared_error),
                            cv=n_folds)

rgf_score = sum(rgf_scores)/n_folds
print('RGF Regressor MSE: {0:.5f}'.format(rgf_score))
rf_score = sum(rf_scores)/n_folds
print('Random Forest Regressor MSE: {0:.5f}'.format(rf_score))

y_pred_rgf = rgf.fit(X_train, y_train).predict(X_test)
y_pred_rf = rf.fit(X_train, y_train).predict(X_test)
