import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

### Title content ###
df = pd.read_csv("/content/drive/MyDrive/datasets/malware-raw-dataset.csv")
st.title("""
Welcome to Team 27's streamlit!

Malware Classification with PE Headers
""")

st.subheader("Sourcing training data-set")
st.caption("Source: https://www.kaggle.com/saurabhshahane/classification-of-malwares")
st.markdown("""
All features are extracted from the header field of PE executable.

**IMAGEDOSHEADER** (19)

"emagic", "ecblp", "ecp","ecrlc","ecparhdr", "eminalloc","emaxalloc","ess","esp", "ecsum","eip","ecs","elfarlc","eovno","eres", "eoemid","eoeminfo","eres2","e_lfanew"

**FILE_HEADER** (7)

"Machine","NumberOfSections","CreationYear","PointerToSymbolTable", "NumberOfSymbols","SizeOfOptionalHeader","Characteristics"

**OPTIONAL_HEADER** (29)

"Magic", "MajorLinkerVersion", "MinorLinkerVersion", "SizeOfCode", "SizeOfInitializedData", "SizeOfUninitializedData", "AddressOfEntryPoint", "BaseOfCode", "BaseOfData", "ImageBase", "SectionAlignment", "FileAlignment", "MajorOperatingSystemVersion", "MinorOperatingSystemVersion", "MajorImageVersion", "MinorImageVersion", "MajorSubsystemVersion", "MinorSubsystemVersion", "SizeOfImage", "SizeOfHeaders", "CheckSum", "Subsystem", "DllCharacteristics", "SizeOfStackReserve", "SizeOfStackCommit", "SizeOfHeapReserve", "SizeOfHeapCommit", "LoaderFlags", "NumberOfRvaAndSizes"

We will be predicting the target variable '**class**' (0 - benign, 1 - malicious)
""")

st.info("More info: https://data.mendeley.com/datasets/xvyv59vwvz/1")

st.header("Malware-raw-dataset details:")
entry_col1, entry_col2 = st.columns(2)
entry_col1.metric("Number of entries (rows)", df.shape[0], delta=None)
entry_col2.metric("Number of features (columns)", df.shape[1], delta=None)
with st.expander("Original entries dataframe"):
  st.dataframe(df)

### Body content ###

#Missing values sanitization
missing_value_fig = plt.figure(figsize=(10, 20))
missing_value_ax = missing_value_fig.add_subplot(1,1,1)

st.subheader("Sanitization of missing values")
st.markdown("""
In data analysis, every dataset that have been collected requires sanitization for better accuracy and efficiency in processing the data.

Hence, we will sanitize the dataset by removing the data columns that are empty, which we will visualize using the missingno library.
""")
st.code("""
missing_value_fig = plt.figure(figsize=(10, 20))
missing_value_ax = missing_value_fig.add_subplot(1,1,1)
""", language="python")

# Create the missing values table
msno.bar(df,color="dodgerblue", sort="descending", ax=missing_value_ax)
with st.expander("Visualization of missing values in dataset"):
  st.write(missing_value_fig)
st.write("The result shows that e_res and e_res2 feature are completely empty, which is the total number of entries in the dataset, this means that these features have no significance and will be dropped.")
st.code("df = df.drop(columns=['e_res', 'e_res2'])", language="python")
df = df.drop(columns=['e_res', 'e_res2'])
st.write("The lower number of features confirms that the insigificant rows are dropped")
missing_col1, missing_col2 = st.columns(2)
missing_col1.metric("Number of entries (rows)", df.shape[0], delta=None)
missing_col2.metric("Number of features (columns)", df.shape[1], delta="-2")

# Visualization on number of benign and malicious executable in the dataset
st.subheader("Benign and malicious executables")
st.markdown("Next, we will visualize the amount of __benign__ and __malicious__ executables in the dataset.")

class_fig = plt.figure(figsize=(10, 8))
class_ax = class_fig.add_subplot(1,2,1)
df['class'].value_counts().plot(kind='bar',figsize=(10, 8), title="Bar chart of 'class' label", color=['Red', 'Green'], ax=class_ax)
class_ax.set_xlabel("Class label")
class_ax.set_ylabel("Counts")
# Plot shows [1, 0] by default. So it is mapped to ['Malicious', 'Benign']
class_ax.set_xticklabels( ('Malicious', 'Benign') )
for p in class_ax.patches:
    class_ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
with st.expander("Diagram of benign and malicious executables"):
  st.write(class_fig)
st.write("As a result, we are able to see that **2683** files are malicious and **2501** files are benign in the dataset.")
st.caption("Referenced source: https://www.shanelynn.ie/bar-plots-in-python-using-pandas-dataframes/")

# Sampling and modelling
st.header("Sampling and modelling of the dataset")
st.markdown("""
We will first obtain a sample of the dataset by randomizing the new set and splitting the data into training and testing sets in a 80/20 ratio respectively.

Then, we decide and create the most optimal model to help predict the data.
""")

#Randomize data by sampling using Pandas' sample function.
st.subheader("Sampling data")
st.write("We can randomize the data by using Panda's sample function. ")
st.code("df_sample = df.sample(frac=1)", language="Python")
df_sample = df.sample(frac=1)
with st.expander("New random sample set of malware-raw-dataset"):
  st.dataframe(df_sample)
st.write("Next, we split the data into 80% for training data and 20% for testing data.")
st.code("""
X = df_sample.iloc[:, 0:53]
y = df_sample.iloc[:, 53]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
""",language="Python")

# split dataset
X = df_sample.iloc[:, 0:53]
y = df_sample.iloc[:, 53]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#Deciding on the model
st.subheader("Deciding on the model")
st.write("""
Since we have a label to predict, the team decided to use classification method (K-Nearest Neighbors, Decision Tree and Random Forest). 

Classification models are being compared and evaluated for its accuracy.
""")
st.markdown("**Modelling**")
st.code("""
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier()))

results = []
names = []

for name, model in models:
	# Performing 10 fold cross validation based on accuracy. 
	kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
""", language="Python")
st.caption("Source: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/")

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier()))

results = []
names = []
accuracy_set = {}

st.write("Result:")
for name, model in models:
	# Performing 10 fold cross validation based on accuracy. 
	kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	accuracy_set[name] = cv_results.mean()
accuracy_df = pd.DataFrame(data=accuracy_set, index=[0])
st.table(accuracy_df)
st.write("With the comparison diagram, the team decided to go with random forest as it had the highest accuracy, as evident from the mean calculation.")

#Visualisation of the model comparison via box-plot
box_plot_fig = plt.figure()
box_plot_fig.suptitle('Algorithm Comparison')
box_plot_ax = box_plot_fig.add_subplot(111)
box_plot_ax.boxplot(results)
box_plot_ax.set_xticklabels(names)
with st.expander("Visualisation of the model comparison (box-plot)"):
  st.write(box_plot_fig)
st.markdown("As the comparison of models returned a random forest of 98% accuracy with default parameters, no parameter tuning was done.")

# Creating the model, training it and predicting the test set
st.subheader("Creating the model, training it and predicting the test set")
st.write("Using random forest modelling: ")
st.code("""
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
""", language="Python")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

st.subheader("Evaluating the Accuracy of the model")
st.write("We used sklearn's library to help evaluate F1 mean and accuracy score of the model.")
accuracy_col1, accuracy_col2 = st.columns(2)
accuracy_col1.metric("F1 score", f1_score(y_test, pred), delta=None)
accuracy_col2.metric("Accuracy score", accuracy_score(y_test, pred), delta=None)

st.subheader("Confusion matrix: Random forest modelling")
st.write("We will also be calculating the confusion matrix to summarize the performance of the random forest model")
st.code("""
confusion_fig, confusion_ax = plt.subplots(figsize=(5.5, 5.5))
cm = confusion_matrix(y_test, pred)
confusion_ax.matshow(cm,  cmap=plt.cm.Blues, alpha=0.30)
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    confusion_ax.text(x=j, y=i,
            s=cm[i, j],
            va='center', ha='center')
    confusion_fig.suptitle('Confusion matrix of Random Forest Model at 98% accuracy ')
    confusion_ax.set_xlabel('Predicted value')
    confusion_ax.set_ylabel('Actual value')
""", language="Python")
st.caption("Reference of confusion matrix plotting: https://www.kaggle.com/marshalltrumbull/malware-prediction-with-97-accuracy-and-f-1-score")
confusion_fig, confusion_ax = plt.subplots(figsize=(5.5, 5.5))
cm = confusion_matrix(y_test, pred)
confusion_ax.matshow(cm,  cmap=plt.cm.Blues, alpha=0.30)
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    confusion_ax.text(x=j, y=i,
            s=cm[i, j],
            va='center', ha='center')
    confusion_fig.suptitle('Confusion matrix of Random Forest Model at 98% accuracy ')
    confusion_ax.set_xlabel('Predicted value')
    confusion_ax.set_ylabel('Actual value')
with st.expander("Confusion diagram"):
  st.write(confusion_fig)

# Top 10 important features
st.subheader("Top 10 important features")
st.write("""One of the key ways to determine a file that is malicious can be seen by its feature. 
Hence, the team finds it crucial to list the top 10 important features of a file to determine if it is a malware.""")
st.code("""
#Create arrays from feature importance and feature names
feature_importance = np.array(rf.feature_importances_)
feature_names = np.array(X_train.columns)
""")
#Create arrays from feature importance and feature names
feature_importance = np.array(rf.feature_importances_)
feature_names = np.array(X_train.columns)

st.write("With the code above, we have obtained the necessary data to plot for our important features, using the seaborn library.")
st.code("""
#Create a DataFrame using a Dictionary
data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)
#Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
#Slice dataframe to top 10
fi_df2 = fi_df.head(10)
#Define size of bar plot and create axes
features_fig = plt.figure(figsize=(10,10))
features_ax = features_fig.add_subplot(1,1,1)
#Plot Searborn bar chart
sns.barplot(x=fi_df2['feature_importance'], y=fi_df2['feature_names'],ax=features_ax)
#Add chart labels
features_fig.suptitle('RANDOM FOREST FEATURE IMPORTANCE')
features_ax.set_xlabel('FEATURE IMPORTANCE')
features_ax.set_ylabel('FEATURE NAMES')
""", language="Python")
#Create a DataFrame using a Dictionary
data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)
#Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
#Slice dataframe to top 10
fi_df2 = fi_df.head(10)
#Define size of bar plot and create axes
features_fig = plt.figure(figsize=(10,10))
features_ax = features_fig.add_subplot(1,1,1)
#Plot Searborn bar chart
sns.barplot(x=fi_df2['feature_importance'], y=fi_df2['feature_names'],ax=features_ax)
#Add chart labels
features_fig.suptitle('RANDOM FOREST FEATURE IMPORTANCE')
features_ax.set_xlabel('FEATURE IMPORTANCE')
features_ax.set_ylabel('FEATURE NAMES')
with st.expander("Top 10 features diagram"):
  st.write(features_fig)

#Conclusion?

### Tail content goes here ###
#References?