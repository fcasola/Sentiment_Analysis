"""
Module reading the configuration file

written by F. Casola, Harvard University - fr.casola@gmail.com
"""
from six.moves import configparser
import json

config = configparser.RawConfigParser()
config.read('../data/Config/config_sentiment.cfg')

#TrainingData
dataset_train_path = config.get("TrainingData", "dataset_train_path")
dataset_test_path = config.get("TrainingData", "dataset_test_path")
N_max_train_test =  json.loads(config.get("TrainingData","N_max_train_test"))

#VecRep
Use_dense_rep = config.getboolean("VecRep", "Use_dense_rep")
tfIdf = config.getboolean("VecRep", "tfIdf")
website_pretrained_Word2vec = config.get("VecRep", "website_pretrained_Word2vec")
Dest_fold_dwl = config.get("VecRep", "Dest_fold_dwl")

#LocalDest
device_name = config.get("LocalDest", "device_name")
pathgraph = config.get("LocalDest", "pathgraph")
path_model = config.get("LocalDest", "path_model")

#TrainingPars
Dim_hidden =  json.loads(config.get("TrainingPars","Dim_hidden"))
learning_rate = config.getfloat("TrainingPars", "learning_rate")
epochs = config.getint("TrainingPars", "epochs")
batch_size = config.getint("TrainingPars", "batch_size")

# create a dictionary
# create a dictionary that stores them all
NLP_dic= {"dataset_train_path": dataset_train_path,
		"dataset_test_path": dataset_test_path,
		"N_max_train_test": N_max_train_test,
		"Use_dense_rep": Use_dense_rep,
		"tfIdf": tfIdf,
		"website_pretrained_Word2vec": website_pretrained_Word2vec,
		"Dest_fold_dwl": Dest_fold_dwl,
		"device_name": device_name,
		"pathgraph": pathgraph,
		"path_model": path_model,
		"Dim_hidden": Dim_hidden,
		"learning_rate": learning_rate,
		"epochs": epochs,
		"batch_size": batch_size}