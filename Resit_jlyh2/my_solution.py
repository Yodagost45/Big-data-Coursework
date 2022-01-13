import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import feature_selection, metrics
from sklearn.model_selection import train_test_split, cross_val_score
def compare_graph(x, y):
	model = smf.ols(formula=y+'~'+x, data=df).fit()
	print(model.params)
	print(model.conf_int())
	model.pvalues
	predicted = model.predict(df[x])
	plt.plot(df[x], df[y], "bo")
	plt.plot(df[x], predicted, "r-", linewidth=2)
	plt.title("linear regression fit")
	plt.xlabel(x)
	plt.ylabel(y)
	plt.show()

def plot_stats(df, col):
    # plot with various axes scales
    fig = plt.figure(1)
    fig.set_size_inches(12,8)
    ## First a box plot
    plt.subplot(2,1,1)
    df.dropna().boxplot(col, vert=False)
    ## Then Plot the histogram 
    plt.subplot(2,1,2)  
    temp = df[col]
    plt.hist(temp, bins = 30)
    plt.ylabel('Number of Datapoints')
    plt.xlabel(col)
    plt.show()
    
def plot_hist(df, col):
    plt.subplot(2,1,2)  
    temp = df[col]
    plt.hist(temp, bins = 30)
    plt.ylabel('Number of Datapoints')
    plt.xlabel(col)
    plt.show()
    
'''This is used to identify outliers and then remove them'''
def identify_outlier(df):
    ## Create a vector of 0 of length equal to the number of rows
    temp = np.zeros(df.shape[0])
    ## test each outlier condition and mark with a 1 as required
    for i, x in enumerate(df['Duration']):
        if (x > 0.7): temp[i] = 1 
    for i, x in enumerate(df['CreditAmount']):
        if (x > 0.5): temp[i] = 1 
    for i, x in enumerate(df['Age']):
        if (x > 0.68): temp[i] = 1 
    for i, x in enumerate(df['ExistingCreditsAtBank']):
        if (x > 0.7): temp[i] = 1 
    df['outlier'] = temp # append a column to the data frame   
    return df

def normalize_df(df, cols):
	for col in cols:
		temp = np.array(df[col])
		temp_norm = (temp-temp.min())/(temp.max()-temp.min())
		df[col] = temp_norm
	return df
	
def categorical_encoding(df):
	#to be called before deleting old columns for Female column, also before normalization
	df["Female"] = np.where(df["SexAndStatus"].str.contains("A92"), 1, 0)
	
	encode_cols = ["CheckingAcctStat", "Savings", "Employment", "Property", "Job", "ForeignWorker"]
	
	for col in encode_cols:
		df[col] = df[col].astype("category").cat.codes
	df = pd.get_dummies(df, columns=["CreditHistory", "Purpose", "Housing"])
	return df
	
def classify_for_threshold(clf, testX, testY, t):
    prob_df = pd.DataFrame(clf.predict_proba(testX)[:, 1])
    prob_df['predict'] = np.where(prob_df[0]>=t, 1, 0)
    prob_df['actual'] = testY
    return pd.crosstab(prob_df['actual'], prob_df['predict'])
    
def logistic_regression(df, no_features_to_select):
	print("****** Logistic Regression with "+str(no_features_to_select)+" features *************")
	model = linear_model.LogisticRegression()
	selector = feature_selection.RFE(model, n_features_to_select=no_features_to_select, step=1)
	X0 = df.loc[:, df.columns != "CreditStatus"]
	Y0 = df["CreditStatus"]
	selector = selector.fit(X0, Y0)
	r_features = X0.loc[:, selector.support_]
	X = r_features
	Y = df["CreditStatus"]
	trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.5, random_state=0)
	clf = linear_model.LogisticRegression()
	clf.fit(trainX, trainY)
	print(' {}'.format(clf.intercept_))
	print(' {}'.format(clf.coef_))
	predicted = clf.predict(testX)
	print("Mean hits: {}".format(np.mean(predicted==testY)))
	scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring='accuracy', cv=8)
	print(scores)
	print("Mean scores: {}".format(scores.mean()))
	xtab = classify_for_threshold(clf, testX, testY, 0.2)
	print("Threshold {}:\n{}\n".format(0.5, xtab))
	print(metrics.classification_report(testY, predicted))
	metrics.plot_roc_curve(clf, testX, testY)
	plt.title("ROC curve and AUC for " + str(no_features_to_select) + " selected features")
	plt.get_current_fig_manager().window.state('zoomed')
	plt.show()

df = pd.read_csv("credit_scores.csv")
cols =  ["CheckingAcctStat", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment", "InstallmentRatePecnt", "SexAndStatus", "OtherDetorsGuarantors", "PresentResidenceTime", "Property", "Age", "OtherInstalments", "Housing", "ExistingCreditsAtBank", "Job", "NumberDependents", "Telephone", "ForeignWorker", "CreditStatus"]
print(df[["CreditAmount", "Duration"]][:3])
'''
print(df.corr(method="kendall"))
print(df.corr(method="pearson"))
print(df.corr(method="spearman"))
print(pd.isnull(df).sum())
'''
num_cols = ["Duration", "CreditAmount", "InstallmentRatePecnt", "PresentResidenceTime", "Age", "ExistingCreditsAtBank", "NumberDependents", "CreditStatus"]
'''
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


print(pd.isnull(df).sum())
df.dropna(axis=0, inplace = True )
print(df.dtypes)
print(df.shape)
'''
print("before")
print(df.describe())
age = np.array(df["Age"])
age_norm = (age-age.min())/(age.max()-age.min())
df = df.assign(Age=age_norm)
print("After")
print(df.describe())
'''
#for col in num_cols:
	#compare_graph(col, "CreditStatus")
	#plot_stats(df, col)
'''
cat_cols =  ["CheckingAcctStat","CreditHistory", "Purpose", "Savings", "Employment", "SexAndStatus", "OtherDetorsGuarantors", "Property", "OtherInstalments", "Housing", "Job", "Telephone", "ForeignWorker"]
'''	
#for col in cat_cols:
#	plot_hist(df, col)
'''

for col in cat_cols:
	print(df.groupby([col, "CreditStatus"]).size())
df = categorical_encoding(df)

compare_graph("Female", "CreditStatus")

print(df.head(10))
	
drop_cols = ["SexAndStatus", "OtherDetorsGuarantors", "OtherInstalments", "Telephone", "PresentResidenceTime", "NumberDependents"]

df = df.drop(columns=drop_cols)

print(df.dtypes)

df = identify_outlier(df)
df = df[df.outlier == 0] # filter for outliers
df.drop('outlier', axis = 1, inplace = True)

for col in drop_cols:
	if col in num_cols:
		num_cols.remove(col)
	if col in cat_cols:
		cat_cols.remove(col)

print(drop_cols)
print(num_cols)
print(cat_cols)

print(df.describe())
df = normalize_df(df, num_cols)
print(df.describe())
'''
for col in num_cols:
	compare_graph(col, "CreditStatus")
	plot_stats(df, col)
'''
number_of_features = list(range(2, len(df.columns)))
for n in number_of_features:
	logistic_regression(df, n)


J_array = []
N_array = list(range(2, len(df.columns)))
for n in N_array:
	print(n)
	model = KMeans(n_clusters=n)
	model.fit(df)


	## J score
	print('J-score = ', model.inertia_)
	J_array.append(model.inertia_)
	#print(' score = ', model.score(df_norm))
	## include the labels into the data
	print(model.labels_)
plt.scatter(N_array, J_array)
plt.xlabel("Number of Clusters")
plt.ylabel("J-Score")
plt.show()
model = KMeans(n_clusters=7)
model.fit(df)


## J score
print('J-score = ', model.inertia_)
#print(' score = ', model.score(df_norm))
## include the labels into the data
print(model.labels_)
pca_data = PCA(n_components=2).fit(df)
pca_2d = pca_data.transform(df)
plt.scatter(pca_2d[:,0], pca_2d[:,1])
plt.title('unclustered data')
plt.show()
plt.scatter(pca_2d[:,0], pca_2d[:,1], c=model.labels_)
plt.title('clustered data')
plt.show()
