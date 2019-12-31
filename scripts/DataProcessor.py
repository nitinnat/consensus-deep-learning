import pandas as pd
import os
import numpy as np
from copy import copy
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging



# constants
DATA_DIR = "../data/"


def load_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--numsplits", type=int, default=10,
	                    help="Number of vertical splits to make on the data.")
	
	parser.add_argument("-d", "--dataset", type=str, 
	                    help="Name of dataset")
	parser.add_argument("-r", "--run", type=int, default=0,
	                    help="Run number, determines random seed.")
	
	args = parser.parse_args()
	return args



def loadData(DATA_DIR, filename):
    """
    Loads csv files where column 0 represents labels.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, filename), header=None)
    df2 = np.array(df)
    labels = df2[:,0]
    df2 = df2[:,1:]
    return df2, labels


def makeBinary(data, labels, classLabel):
    """
    Makes the dataset binary by changing the given class label to 1. 
    """
    zippedData = list(zip(data, labels))
    zippedData = [[dataPoint, 1] if label == 8 else [dataPoint, 0] for dataPoint, label in zippedData]
    data, labels = list(zip(*zippedData))
    data, labels = np.array(data), np.array(labels)
    assert data.shape[0] == labels.shape[0]
    print(data.shape, labels.shape)
    return data, labels

def makeBinaryBalanced(data, labels, posClassLabel, negClassLabel):
    """
    Makes the dataset binary and balanced by changing the posClassLabel class to 0, and the negClassLabel class to 1.
    """
    zippedData = list(zip(data, labels))
    zippedData = [[dataPoint, 1] if label == posClassLabel else [dataPoint, 0] for dataPoint, label in zippedData 
                  if label in [posClassLabel, negClassLabel]]
    
    data, labels = list(zip(*zippedData))
    data, labels = np.array(data), np.array(labels)
    assert data.shape[0] == labels.shape[0]
    print(data.shape, labels.shape)
    return data, labels


def split_padded(matrix,labels, n):
    a = np.arange(matrix.shape[1])
    padding = (-len(a))%n
    index_arrays = np.split(np.concatenate((a,np.zeros(padding))).astype(int),n)
    index_arrays[-1] = np.trim_zeros(index_arrays[-1] , "b")
    matrices = [np.hstack((labels[:,np.newaxis], matrix[:, index_arrays[i]])) for i in range(len(index_arrays))]
    return matrices


def verticalPartition(data, labels):
    assert round(sum(probVector),3) == 1
    df = copy(data)
    numFeatures = data.shape[1]
    splitDfs = []
    for i in range(len(probVector)):
        numFeats = probVector[i]*numFeatures
        tempDf = df[:,0:min(int(numFeats), df.shape[1]-1)]
        tempDf = np.hstack((labels[:,np.newaxis], tempDf))
        splitDfs.append(tempDf)
        df = df[:, int(numFeats)+1:]
    return splitDfs


def saveSplitFiles(DATA_DIR, baseFilename, splitDfs):
    """
    Saves files into respective CSV files.
    """
    for i in range(len(splitDfs)):
        temp_df = splitDfs[i]
        temp_filename = baseFilename.split(".")[0] + "_" + str(i) + ".csv"        
        temp_df = pd.DataFrame(data=temp_df, index=None)
        temp_df.to_csv(os.path.join(DATA_DIR, temp_filename), index=False, header=False)
        print("File saved in {}".format(os.path.join(DATA_DIR, temp_filename)))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def process_sonar(num_splits):

	dataset_dir = os.path.join(DATA_DIR, "sonar")
	trainFilename = "sonar_train.csv"
	testFilename = "sonar_test.csv"

	# Process training set
	data, labels = loadData(dataset_dir, trainFilename)
	print(data.shape, labels.shape)
	splitDfs = split_padded(data, labels, num_splits)
	print([a.shape for a in splitDfs])
	saveSplitFiles(dataset_dir, trainFilename, splitDfs)
	print(Counter(labels))
	df_to_save = pd.DataFrame(np.hstack((labels.reshape(-1,1), data)))
	print(df_to_save.shape)
	df_to_save.to_csv(os.path.join(dataset_dir, trainFilename.split(".")[0] + "_binary.csv"), index=False, header=False)



	# Process test set
	data, labels = loadData(dataset_dir, testFilename)
	splitDfs = split_padded(data, labels, num_splits)
	print([a.shape for a in splitDfs])
	saveSplitFiles(dataset_dir, testFilename, splitDfs)
	print(Counter(labels))

	df_to_save = pd.DataFrame(np.hstack((labels.reshape(-1,1), data)))
	print(df_to_save.shape)
	df_to_save.to_csv(os.path.join(dataset_dir, testFilename.split(".")[0] + "_binary.csv"), index=False, header=False)

	print("Sonar data available in {}".format(dataset_dir))



def process_madelon(num_splits):

	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import MinMaxScaler

	dataset_dir = os.path.join(DATA_DIR, "madelon")

	import datapackage
	data_url = 'https://datahub.io/machine-learning/madelon/datapackage.json'
	# to load Data Package into storage
	package = datapackage.Package(data_url)
	# to load only tabular data
	resources = package.resources
	for resource in resources:
	    if resource.tabular:
	        madelon_data = pd.read_csv(resource.descriptor['path'])


	X_madelon, y_madelon = madelon_data.iloc[:, :-1], madelon_data.iloc[:, -1]
	X_madelon = np.array(X_madelon)
	y_madelon = np.array(y_madelon) 
	X_madelon = MinMaxScaler().fit_transform(X_madelon) # scale
	y_madelon[y_madelon == 1] = 0
	y_madelon[y_madelon == 2] = 1
	X_madelon_train, X_madelon_test, y_madelon_train, y_madelon_test = train_test_split(X_madelon, y_madelon, test_size=0.2, random_state=42)

	all_train = np.hstack((y_madelon_train.reshape(-1,1), X_madelon_train))
	all_test = np.hstack((y_madelon_test.reshape(-1,1), X_madelon_test))
	
	pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "madelon_train_binary.csv"), index=None, header=None)
	pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "madelon_test_binary.csv"), index=None, header=None)
	
	trainFilename = "madelon_train.csv"
	testFilename = "madelon_test.csv"
	# Load and process synthetic data
	# Process training set
	splitDfs = split_padded(X_madelon_train, y_madelon_train, num_splits)
	print([a.shape for a in splitDfs])
	saveSplitFiles(dataset_dir, trainFilename, splitDfs)
	print(Counter(y_madelon_train))


	# Process test set
	splitDfs = split_padded(X_madelon_test, y_madelon_test, num_splits)
	print([a.shape for a in splitDfs])
	saveSplitFiles(dataset_dir, testFilename, splitDfs)
	print(Counter(y_madelon_test))

	print("madelon data available in {}".format(dataset_dir))

if __name__ == "__main__":
	args = load_args()
	
	if args.dataset == "sonar":
		process_sonar(args.numsplits)

	elif args.dataset == "madelon":
		process_madelon(args.numsplits)