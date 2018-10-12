import pandas as pd
import numpy as np
from pprint import pprint
import random
import pickle


file = open("Dataset.txt", "r")
data = list()
for line in file:
    data.append(line.split(','))
file.close()
random.shuffle(data)
train_data = data[:int((len(data) + 1) * .80)]  # Remaining 80% to training set
test_data = data[int(len(data) * .80 + 1):]  # Splits 20% data to test set

df_train = pd.DataFrame(train_data)
df_train.to_csv("train_dataset.csv")
dataset_train = pd.read_csv('train_dataset.csv',
                      names=['animal_name','hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','class',])
dataset_train=dataset_train.drop('animal_name',axis=1)

df_test = pd.DataFrame(test_data)
df_test.to_csv("test_dataset.csv")
dataset_test = pd.read_csv('test_dataset.csv',
                      names=['animal_name','hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','class',])
dataset_test=dataset_test.drop('animal_name',axis=1)

#traing the classifier

def entropy(target):
    elements,counts = np.unique(target,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name="class"):
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    #Calculate the values and the corresponding counts for the split attribute
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)

    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):

    #stopping criteria --> If one of this is satisfied, return a leaf node#

    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]

    #for feature space empty= mode target feature value of the direct parent node
    #for direct parent node = node which has  current run of the ID3 algorithm
    #the mode target feature value stored in the parent_node_class variable.

    elif len(features) ==0:
        return parent_node_class

    #If none of the above holds true, grow the tree!

    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]

        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}


        #Removing  feature with the best inforamtion gain
        features = [i for i in features if i != best_feature]

        #Grow a branch under the root node

        for value in np.unique(data[best_feature]):
            value = value
            #Spliting the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            #Calling  ID3 algorithm
            subtree = ID3(sub_data,dataset_train,features,target_attribute_name,parent_node_class)

            #Adding subtree
            tree[best_feature][value] = subtree
    return(tree)

def predicct(query,tree,default = 1):

    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]]
            except:
                return default

            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predicct(query,result)
            else:
                return result


def test(testdata,tree):
    #Creating new query instances by removing target feature column
    queries = testdata.iloc[:,:-1].to_dict(orient = "records")
    #Creating empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    #Calculate the prediction accuracy
    for i in range(len(testdata)):
        predicted.loc[i,"predicted"] = predicct(queries[i],tree,1.0)
        #a=(np.sum(predicted["predicted"] == testdata["class"])/len(testdata))*100
        #type(a)
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == testdata["class"])/len(testdata))*100,'%')
    a=(np.sum(predicted["predicted"] == testdata["class"])/len(testdata))*100
    return a  
     
def testflask(testdata,tree):
    
        predicted = predicct(testdata,tree,1.0)
        return predicted
    
training_data = dataset_train.reset_index(drop=True)
testing_data=dataset_test.reset_index(drop=True)


trainedtree = ID3(training_data,training_data,training_data.columns[:-1])
t=test(testing_data,trainedtree)
print(t)

pickle.dump(trainedtree, open("model.pkl" , "wb"))


