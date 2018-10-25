# A bunch of lines to run on some new data that you plan to run some regression on

# Descriptive statistics
df.describe()  
df.y.describe()
df[list_of_columns].describe()  # don't mix numerical and categorical

# Nulls in data
int(df.y.isnull().any())

# Normality checks, fit distribution to data histogram
sns.distplot(df.y, fit=stats.norm)
fig = plt.figure()
res = stats.probplot(df.y, plot=plt)

#  Explore relationship between vars
corrmat = df.corr()
fig = plt.figure()
sns.heatmap(corrmat, annot=True, fmt='.2f', square=True)

# Feature transformations
log_transformed = np.log(df[column])
exp_transformed = np.exp(df[column])
squared = df[column]**2
cubed = df[column]**3
tanh = np.tanh(df[column])
# Test the corr
r = np.corrcoef(df.y, log_transformed)[0][1]

# Explore categoricals
for column in CATEGORICAL_NAMES:
    sns.boxplot(x=column, y="y", data=df)
    plt.show()

# Explore the connection between y and predictors
for column in NUMERIC_NAMES:
    sns.scatterplot(x=column, y="y", data=df, size=8)    
    plt.show()

# Zscore numericals
scaler = StandardScaler()
zscored_df = pd.DataFrame(scaler.fit_transform(df[NUMERIC_NAMES]), columns=NUMERIC_NAMES)

# Check for PCA
pca.fit(zscored_df_num)
pca.explained_variance_ratio_

# Encode categoricals ordinally (might introduce aritificial order info)
ordinal_encoded_df_cat = pd.DataFrame()
label_encoder = LabelEncoder()
for column in CATEGORICAL_NAMES:
    ordinal_encoded_df_cat[column] = label_encoder.fit_transform(df[column])

# Encode categoricals with one-hot
label_bin_df_cat = pd.DataFrame()
label_binarizer = LabelBinarizer()
for i, column in enumerate(CATEGORICAL_NAMES):
    col_encoded = pd.DataFrame(label_binarizer.fit_transform(df[column]), 
                               columns=[column + "_" + l for l in label_binarizer.classes_])
    label_bin_df_cat = pd.concat([label_bin_df_cat, col_encoded], axis=1)

# Mass evaluation
DATASETS = {"name": data_df}
model_scores = dict()    
ESTIMATORS = {"XGBoost": xgb.XGBRegressor()}

for ds_name, ds in DATASETS.items():
    X_train, X_test, y_train, y_test = train_test_split(ds, df_train["y"], test_size=0.33)
    model_scores[ds_name] = {}
    for model_name, model in ESTIMATORS.items():
        print(ds_name, model)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_hat))
        r2 = r2_score(y_test, y_hat)
        model_scores[ds_name][model_name] = (rmse, r2)
        print(rmse, r2)
        try:
            feature_importances = pd.DataFrame(
                model.feature_importances_, 
                index = X_train.columns, 
                columns=['importance']
            ).sort_values('importance', ascending=False)
            print(feature_importances)
        except:
            print("No importances for model: {}".format(model_name))
# Best model
best_model_per_dataset = {}
for ds_name, model_dict in model_scores.items():
    t_r2 = -1
    t_rmse = np.inf
    for model_name, (rmse, r2) in model_dict.items():
        if rmse < t_rmse:
            t_rmse = rmse
            best_model_per_dataset[ds_name] = (model_name, rmse, r2)

# Predicted vs Observed
y_hat_train = model.predict(train_dataset)
sns.scatterplot(x=df.y, y=y_hat_train)    
plt.show()
