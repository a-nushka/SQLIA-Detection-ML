import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

import csv;
import time;
from io import StringIO;

log_file="C:\ProgramData\MySQL\MySQL Server 8.3\Data\LAPTOP-SCQ81L58.txt"


data=pd.read_csv("Modified_SQL_Dataset.csv")
test_data  = pd.read_csv("Modified_SQL_Dataset.csv")

# data.shape
print(test_data.shape)




def convert_To_Lower_case(x):
    return x.lower()

def preprocess(data):
    data['Query'] = data['Query'].apply(convert_To_Lower_case)
    boolean = data.duplicated(subset = ['Query','Label'])
    data.drop_duplicates(subset = ['Query','Label'],inplace = True)
    #dropping the few queries which has both the label to avoid ambiguity
    data.drop_duplicates(subset = ['Query'],keep = False,inplace = True)
    return data


data=preprocess(data)
boolean = data.duplicated(subset = ['Query','Label'])
test_data.drop_duplicates(subset = ['Query','Label'],inplace = True)
    #dropping the few queries which has both the label to avoid ambiguity
test_data.drop_duplicates(subset = ['Query'],keep = False,inplace = True)



def tokenize_query(query):
    return nltk.word_tokenize(query)


def count_single_quotes(tokenized_row):
    return sum(1 for token in tokenized_row if "'" in token)

def count_double_quotes(tokenized_row):
    return sum(1 for token in tokenized_row if '"' in token)

def count_parentheses(tokenized_row):
    count_open = sum(1 for token in tokenized_row if "(" in token)
    count_close = sum(1 for token in tokenized_row if ")" in token)
    return count_open, count_close

def count_underscores(tokenized_row):
    return sum(1 for token in tokenized_row if "_" in token)

def count_hexadecimal_numbers(tokenized_row):
    return sum(1 for token in tokenized_row if token.startswith("0x") or token.startswith("0X"))

def count_commas(tokenized_row):
    return sum(1 for token in tokenized_row if "," in token)

def count_white_spaces(tokenized_row):
    return sum(1 for token in tokenized_row if token.isspace() or token == '')

def count_logical_operators(tokenized_row):
    logical_operators = ['and', 'or', 'not', 'xor']
    return sum(1 for token in tokenized_row if token.lower() in logical_operators)

def count_single_line_comments(tokenized_row):
    count = 0
    in_comment = False
    for token in tokenized_row:
        if token.startswith('#'):
            count += 1
        elif '#' in token:
            count += 1
            in_comment = True
        elif in_comment and token.endswith('#'):
            count += 1
            in_comment = False
        elif in_comment:
            count += 1
    return count

def create_features(data):
  data['Num_Single_Quotes'] = data['tokenized_queries'].apply(count_single_quotes)
  data['Num_Double_Quotes'] = data['tokenized_queries'].apply(count_double_quotes)
  data['Num_Open_Parentheses'], data['Num_Close_Parentheses'] = zip(*data['tokenized_queries'].apply(count_parentheses))
  data['Num_Underscores'] = data['tokenized_queries'].apply(count_underscores)
  data['Num_Hexadecimal_Numbers'] = data['tokenized_queries'].apply(count_hexadecimal_numbers)
  data['Num_Commas'] = data['tokenized_queries'].apply(count_commas)
  data['Num_White_Spaces'] = data['tokenized_queries'].apply(count_white_spaces)
  data['Num_Logical_Operators'] = data['tokenized_queries'].apply(count_logical_operators)
  data['Num_Single_Line_Comments'] = data['tokenized_queries'].apply(count_single_line_comments)
  return data


data['tokenized_queries'] = data['Query'].apply(tokenize_query)


data = create_features(data)


test_data['tokenized_queries'] = test_data['Query'].apply(tokenize_query)


test_data = create_features(test_data)

data.to_csv("Preprocessed_data.csv",index = False)
test_data.to_csv("Preprocessed_test_data.csv",index=False)


data = pd.read_csv("Preprocessed_data.csv")
test_data = pd.read_csv("Preprocessed_test_data.csv")


Y_train = data['Label']
data = data.drop(['Label'],axis =1 )
Y_test = test_data['Label']
test_data = test_data.drop(['Label'],axis = 1)
#print(Y_test)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,2),max_features = 50000)
X = vectorizer.fit(data['Query'])
data_query = X.transform(data['Query'])
test_data_query = X.transform(test_data['Query'])


len(X.vocabulary_)

# test_data.head(5)

data = data.drop(['Query','tokenized_queries'],axis =1 )
test_data = test_data.drop(['Query','tokenized_queries'],axis =1 )

from scipy.sparse import hstack
data = hstack((data,data_query)).tocsr()
test_data = hstack((test_data,test_data_query)).tocsr()
#print(data)


from sklearn import preprocessing
scaler = preprocessing.StandardScaler(with_mean=False).fit(data)
data = scaler.transform(data)

scaler = preprocessing.StandardScaler(with_mean=False).fit(test_data)
test_data = scaler.transform(test_data)




# Instantiate a RandomForestClassifier with reduced complexity
rf_model_reduced = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

# Fit the reduced complexity model to the training data
rf_model_reduced.fit(data, Y_train)



tree = rf_model_reduced.estimators_[0]

# Apply pruning to the tree (example: post-pruning with cost-complexity pruning)
pruned_tree = DecisionTreeClassifier(ccp_alpha=0.01)  # Adjust ccp_alpha as needed
pruned_tree.fit(data, Y_train)

# Assuming test_data is your test dataset
# Make predictions using the pruned tree
y_pred_pruned = pruned_tree.predict(test_data)

# Evaluate the pruned tree's performance
precision_pruned = precision_score(Y_test, y_pred_pruned)
recall_pruned = recall_score(Y_test, y_pred_pruned)
f1_pruned = f1_score(Y_test, y_pred_pruned)
accuracy = accuracy_score(Y_test, y_pred_pruned)
print("Accuracy:", accuracy)
print("Precision (Pruned Tree):", precision_pruned)
print("Recall (Pruned Tree):", recall_pruned)
print("F1 Score (Pruned Tree):", f1_pruned)

while True:
    with open(log_file, 'r') as file:
        # Move the file pointer to the end of the file
        file.seek(0, 2)

        # Get the current position of the file pointer
        current_position = file.tell()

        # Move to the beginning of the last line
        file.seek(max(current_position - 76,0))  # Set buffer size based on expected line length

        # Read the last line from the file
        last_line = file.readline()
        query_log = pd.DataFrame([last_line], columns=['Query'])

        
        # query_log['Query'] = query_log['Query'].apply(convert_To_Lower_case)
        query_log['tokenized_queries'] = query_log['Query'].apply(tokenize_query)
        query_log = create_features(query_log)
        query_log.to_csv("Preprocessed_query.csv",index=False)
        query_log = pd.read_csv("Preprocessed_query.csv")
        query_log_vector = X.transform(query_log['Query'])
        query_log = query_log.drop(['Query','tokenized_queries'],axis =1 )
        
       
        query_log = hstack((query_log,query_log_vector)).tocsr()
       
        query_log = scaler.transform(query_log)
        
        predictions = pruned_tree.predict(query_log)
        if(predictions[0]==1):
         print("Malicious Sql query")
        else:
          print(" Sql Query")

        
       

    time.sleep(3)