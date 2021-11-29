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
import requests

### Title content ###
st.title("""
Welcome to Team 27's Dashboard!

Malware Classification with PE Headers
""")

st.header("Sourcing training data-set")
st.caption("Source: https://www.kaggle.com/saurabhshahane/classification-of-malwares")
with st.expander("Features information"):
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

st.write(":heavy_minus_sign:" * 34)

### Importing dataset from the author's github ###
url = "https://raw.githubusercontent.com/urwithajit9/ClaMP/master/dataset/ClaMP_Raw-5184.csv"
r = requests.get(url, allow_redirects=True)

open('ClaMP_Raw-5184.csv', 'wb').write(r.content)
df = pd.read_csv("ClaMP_Raw-5184.csv")

st.header("Understanding the data")
### Show first 5 rows, giving a brief insight into dataset ###
st.dataframe(df.head())
entry_col1, entry_col2 = st.columns(2)
### Showing number of rows and columns (entries and features) ###
entry_col1.metric("Number of entries (rows)", df.shape[0], delta=None)
entry_col2.metric("Number of features (columns)", df.shape[1], delta=None)
### User can expand to see all the entries if desired ###
with st.expander("Click to view all the entries"):
  st.dataframe(df)

### Body content ###
st.subheader("Visualisation of NULL values")

### Preparing the axis to plot ### 
missing_value_fig = plt.figure(figsize=(10, 20))
missing_value_ax = missing_value_fig.add_subplot(1,1,1)

### Plotting the missing values on the axis ### 
msno.bar(df,color="dodgerblue", sort="descending", ax=missing_value_ax)
### Writing to the dashboard ###
st.write(missing_value_fig, height=5)

st.subheader("Features changes after dropping null values")

### Dropping the NULL columns ### 
df = df.drop(columns=['e_res', 'e_res2'])
### Showing the first 5 rows after dropping of columns ###
st.dataframe(df.head())
missing_col1, missing_col2 = st.columns(2)
### Showing number of rows and columns (entries and features) after dropping
missing_col1.metric("Number of entries (rows)", df.shape[0], delta=None)
missing_col2.metric("Number of features (columns)", df.shape[1], delta="-2")
### User can expand to see all the entries after changes if desired ###
with st.expander("Click to view entries post-sanitization"):
  st.dataframe(df)

# Visualization on number of benign and malicious executable in the dataset
st.subheader("Visualisation of label classifications in dataset")
# Preparing the plot 
class_fig = plt.figure(figsize=(10, 8))
class_ax = class_fig.add_subplot(1,2,1)
# Generating the bar chart based on the 'class' feature; the label
df['class'].value_counts().plot(kind='bar',figsize=(10, 8), title="Bar chart of 'class' label", color=['Red', 'Green'], ax=class_ax)
# Setting the name of the X and Y axis
class_ax.set_xlabel("Class label")
class_ax.set_ylabel("Counts")
# Plot shows [1, 0] by default, so it is mapped to ['Malicious', 'Benign']
class_ax.set_xticklabels( ('Malicious', 'Benign') )
# Display the numbers of malicious and benign executable on the bar-chart + beautification
for p in class_ax.patches:
    class_ax.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# Write to dashboard
st.write(class_fig)
st.caption("Referenced source: https://www.shanelynn.ie/bar-plots-in-python-using-pandas-dataframes/")

# Sampling and modelling
st.subheader("Randomizing the dataset")

#Randomize data by sampling using Pandas' sample function.
st.write("As the original dataset was sorted, it would affect the model training. Hence, dataset will be randomized.")
# Shuffling the dataset
df_sample = df.sample(frac=1, random_state=27)
# Show the first 5 rows after randomization
st.dataframe(df_sample.head())
# Optionally allow user to view all entries after shuffling if desired
with st.expander("Click to view the shuffled entries"):
  st.dataframe(df_sample)

# Splitting the dataset into 80% training and 20% testing
X = df_sample.iloc[:, 0:53]
y = df_sample.iloc[:, 53]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

st.write(":heavy_minus_sign:" * 34)
#Deciding on the model
st.header("Deciding on the model using classification")
st.caption("Source: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/")

# Creating the model to be compared
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier()))

results = []
names = []
accuracy_set = {}

# Performing 10 fold cross validation with scoring based on accuracy. 
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	accuracy_set[name] = cv_results.mean()
# Saving the means result to a dataframe to be displayed in a table later on. 
accuracy_df = pd.DataFrame(data=accuracy_set, index=[0])

#Visualisation of the model comparison via box-plot
box_plot_fig = plt.figure()
box_plot_fig.suptitle('Algorithm Comparison')
box_plot_ax = box_plot_fig.add_subplot(111)
box_plot_ax.boxplot(results)
box_plot_ax.set_xticklabels(names)
st.write(box_plot_fig)
# Write the means result to a tabular view in the dashboard.
st.table(accuracy_df)
st.markdown("**Random Forest** model was chosen as it has the highest accuracy and is not wide spread.")

st.write(":heavy_minus_sign:" * 34)
# Creating the model, training it and predicting the test set
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

st.subheader("Evaluation of output prediction based on F1 mean and accuracy score")
accuracy_col1, accuracy_col2 = st.columns(2)
# Displaying the F1 and accuracy score in a metric view 
accuracy_col1.metric("F1 score", f1_score(y_test, pred), delta=None)
accuracy_col2.metric("Accuracy score", accuracy_score(y_test, pred), delta=None)

st.subheader("Confusion matrix visualisation")
st.caption("Reference of confusion matrix plotting: https://www.kaggle.com/marshalltrumbull/malware-prediction-with-97-accuracy-and-f-1-score")

# Gets the confusion matrix result
cm = confusion_matrix(y_test, pred)
# Preparing the confusion matrix plot
confusion_fig, confusion_ax = plt.subplots(figsize=(5.5, 5.5))
# Styling the confusion matrix
confusion_ax.matshow(cm,  cmap=plt.cm.Blues, alpha=0.30)
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    confusion_ax.text(x=j, y=i,
            s=cm[i, j],
            va='center', ha='center')
    confusion_fig.suptitle('Confusion matrix of Random Forest Model at 98% accuracy ')
    confusion_ax.set_xlabel('Predicted value')
    confusion_ax.set_ylabel('Actual value')
# Writing confusion matrix to dashboard
st.write(confusion_fig)

# Top 10 important features
st.subheader("Visualisation of the top 10 most important features")
#Create arrays from feature importance and feature names
feature_importance = np.array(rf.feature_importances_)
feature_names = np.array(X_train.columns)

# Create a DataFrame using a Dictionary
data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)
# Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
# Slice dataframe to top 10
fi_df2 = fi_df.head(10)
# Define size of bar plot and create axes
features_fig = plt.figure(figsize=(10,10))
features_ax = features_fig.add_subplot(1,1,1)
# Plot Searborn bar chart
sns.barplot(x=fi_df2['feature_importance'], y=fi_df2['feature_names'],ax=features_ax)
# Add chart labels
features_fig.suptitle('RANDOM FOREST FEATURE IMPORTANCE')
features_ax.set_xlabel('FEATURE IMPORTANCE')
features_ax.set_ylabel('FEATURE NAMES')
# Writes feature importance graph to dashboard
st.write(features_fig)
