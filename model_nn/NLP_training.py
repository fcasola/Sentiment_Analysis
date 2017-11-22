"""
@author: Francesco Casola, Harvard University
fr.casola@gmail.com

Training module for the NLP analysis of Amazon reviews
"""
# run nltk.download() to install all corpora dependencies
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string, re, os, h5py, urllib, progressbar, bz2, warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import tensorflow as tf
import gensim as gs
import numpy as np
from collections import Counter
# comparing the model with a standard logistic regression in sklearn
import sklearn.linear_model as sklr
# personal modules
from NLP_config import *

# Reduce verbosity of tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def Loading_dataset(path_training,path_test,N_max_train_test):
    '''
    Module for loading the dataset
    '''
    # loading dataset...
    # initialize positive and negative sentences 
    label_dic = {0: '__label__1',1: '__label__2'}
    sent_train = []
    sent_test = []
    y_train = []
    y_test = []   
    # training
    print('Loading training dataset')
    with bz2.open(path_training, "rt", encoding = "UTF-8") as bz_file:
        for i,Sent in enumerate(bz_file):
            if i>=N_max_train_test[0]:
                break
            else:
                if label_dic[0] in Sent.split(' '):
                    sent_train.append(Sent.replace(label_dic[0],''))
                    y_train.append(0)
                elif label_dic[1] in Sent.split(' '):
                    sent_train.append(Sent.replace(label_dic[1],''))
                    y_train.append(1)
                else:
                    warnings.warn("Review contains no label!")
    print('Done!')    
    # test
    print('Loading validation dataset')
    with bz2.open(path_test, "rt", encoding = "UTF-8") as bz_file:
        for i,Sent in enumerate(bz_file):
            if i>=N_max_train_test[1]:
                break
            else:
                if label_dic[0] in Sent.split(' '):
                    sent_test.append(Sent.replace(label_dic[0],''))
                    y_test.append(0)
                elif label_dic[1] in Sent.split(' '):
                    sent_test.append(Sent.replace(label_dic[1],''))
                    y_test.append(1)
                else:
                    warnings.warn("Review contains no label!")
    print('Done!')            
    # return data
    return sent_train,sent_test,y_train,y_test

def cleaning_doc(list_doc):
    '''
    Module for noise removal and text cleaning:
        the module removes stopwords (common words in english language),
        punctuation and applies lemmatization.
        
        - cleaning_doc(list_doc)
            The input list_doc contains a list of sentences that need to be cleaned 
    '''
    # common words in english language
    stopw = set(stopwords.words('english'))
    # punctuation
    exclp = set(string.punctuation)
    # additional english expressions like
    # saxon genitive, name with initials, dashes etc..
    lookup = {'\\\'s\s': ' ','\s\w{1}\s\.' : ' ', '--' : ' ', '-' : ' ', '\\\'em\s': ' ', \
              'i\'d': ' ','\"\w+\'\"': '', '\n': '', "\.\s": ' '}  
    if __name__ == "__main__":
        print('Removing the noise from the text. Pls. wait..')
    # Start to lemmatize
    lemma = WordNetLemmatizer()
    # Iterate over the sentences
    for numSent,Sent in enumerate(list_doc):
        # put lowercase
        Sent = Sent.lower()
        # remove expressions in lookup table
        for lkexp in lookup.keys():
                find_oc = re.finditer(lkexp,Sent)
                # replace occurrences with lookup table values
                for sub in find_oc:
                    Sent = re.sub(lkexp,lookup[lkexp],Sent)
        # remove punctuation
        for ch in exclp:
            Sent = Sent.replace(ch,' ')        
        # proceed with other kind of cleanings
        cleaned_str = []
        # remove stopwords
        cleaned_str = [word for word in Sent.split() if word not in stopw]
        # lemmatize the remaining set
        cleaned_str = [lemma.lemmatize(word) for word in cleaned_str]
        # store the changes in the mutable list
        list_doc[numSent] = ' '.join(cleaned_str).strip()
    if __name__ == "__main__":
        print('Done!')    

def Word_representation(dataset_A,dataset_B,Use_dense_rep,tfIdf,siteWord2vec,filename,pathgraph):        
    '''
    Module to create a vectorial representation for the training data    
    
    Word_representation(dataset_A,dataset_B,Use_dense_rep,tfIdf,siteWord2vec,filename)
        -dataset_A: training set
        -dataset_B: validation set
        -Use_dense_rep: Boolean. True when a dense representation is used.
            If False, pls specify whether to use tfIdf for counting or not.
        -siteWord2vec: site where to find the Word2Vec file (if needed)
        -filename: full filename where to find the Word2Vec file on the local drive
        -pathgraph: A full dictionary is saved at destination pathgraph for later use in prediction
    '''
    # We start creating a comprehensive review list
    tot_list_data = [doc.split() for doc in dataset_A+dataset_B]
    # Creating labels
    N_A = len(dataset_A)
    N_B = len(dataset_B)
    # Representation of the dictionary in vectorial form
    Dictionary_rep = {}
    ## loading dense representation
    model = None        
    if Use_dense_rep:
        # check if google Word2Vec file exists
        if os.path.isfile(filename.strip()): 
            # loading the representation 
            print('Loading the Word2Vec database (1.5Gb file!). Needs a 64 bit Python version')
            model = gs.models.KeyedVectors.load_word2vec_format(filename, binary=True)  
            print('Done')
        else:            
            print('Downloading google word2vec from the web. Pls wait...')
            urllib.request.urlretrieve(siteWord2vec, filename)        
            print('Done with downloading. Now loading word2vec..')
            model = gs.models.KeyedVectors.load_word2vec_format(filename, binary=True)  
            print('Done')
        # now creating a matrix using the dense representation
        # getting the vocabulary
        word_vectors = model.wv
        # Initializing training matrix
        print('Constructing a dictionary')
        X_training = np.zeros((N_A+N_B,model.vector_size))
        counts_mis = Counter()
        counts_tot = Counter()
        for i,Sent in enumerate(dataset_A+dataset_B):
            for word in Sent.split(' '):
                counts_tot[word]+=1
                if word in word_vectors.vocab:
                    Dictionary_rep[word] = model[word]
                    X_training[i,:] = np.add(X_training[i,:], model[word])            
                else:
                    counts_mis[word]+=1
        print('Training matrix created. %d words out of %d were not in Google dictionary'%(len(counts_mis),len(counts_tot)))
    else:
        if tfIdf:
            # using Tfid to count occurrencies
            print('Building the training matrix')
            try:
                obj = TfidfVectorizer()
                corpus = dataset_A+dataset_B
                X_training = obj.fit_transform(corpus)
                X_training = X_training.todense()
                print('Done')
            except:
                raise Exception('Sparse dictionary representation too big, use a dense one!')                
        else:
            # build a vocabulary
            print('Building a vocabulary')
            dictionary = gs.corpora.Dictionary(tot_list_data)         
            #construct 1-hot vectors starting from a   
            newdic = dict((v, k) for k, v in dict(dictionary).items())
            try:
                X_training = np.zeros((N_A+N_B,len(newdic)))
                print('Building the training matrix')
                for i,j in enumerate(tot_list_data):
                    for p in j:
                        onehotvec = np.zeros((len(newdic)))
                        onehotvec[newdic[p]] = 1
                        Dictionary_rep[p] = onehotvec
                        X_training[i,newdic[p]] += 1   
                print('Done')
            except:
                raise Exception('Sparse dictionary representation too big, use a dense one!')                                
    # Save the dictionary to file for use during prediction
    print('Saving a smaller dictionary to file for use at prediction time.')
    save_to_hdf5(Dictionary_rep,  os.path.join(pathgraph,'dictionary_model.h5'))            
    # return the training matrix 
    return X_training[0:N_A],X_training[N_A:]

def save_to_hdf5(dic, filename):
    """
    Compact way to save to hierarchical data format, Part I
    
    Input parameters are
        *dic: Dictionary, in the form {key: numerical or string item}
        *filename: full *.h5 name of the file to save 
        
    """
    with h5py.File(filename, 'w') as h5file:
        save_dict_contents(h5file, '/', dic)

def save_dict_contents(h5file, path, dic):
    """
    Compact way to save to hierarchical data format, Part II
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        else:
            raise ValueError('Cannot save %s type'%type(item))

def neural_net_model(Dim_net,learning_rate):
    '''
    Module defining the model for the neural net
    '''
    seed = 128
    
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, Dim_net[0]], name='input_net')
    y = tf.placeholder(tf.float32, [None, Dim_net[-1]], name='target_label')
    # initialize enough corresponding hidden layers and connect them
    tf_variables = {}
    layers = [x]
    for i,j in enumerate(Dim_net[0:-1]):
        matrix_name = 'hidden_m'+str(i)
        bias_name = 'hidden_b'+str(i)
        tf_variables[matrix_name] = tf.Variable(tf.random_normal([j,Dim_net[i+1]],seed=seed), name=matrix_name)
        tf_variables[bias_name] = tf.Variable(tf.random_normal([Dim_net[i+1]],seed=seed), name=bias_name)
        ## connect the layers
        #### we connect all layers with ReLu, except the last one which is a softmax for classification
        if i==(len(Dim_net[0:-1])-1):
            # softmax applied automatically in the loss definition
            layers.append(tf.add(tf.matmul(layers[i],tf_variables[matrix_name]),tf_variables[bias_name]))
        else:
            layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i],tf_variables[matrix_name]),tf_variables[bias_name])))
    ## define cost function 
    ## We put output as [0,1] from 0,1, meaning an Nx2 matrix instead of Nx1
    ## it is all well explained here:
    ## https://stackoverflow.com/questions/35277898/tensorflow-for-binary-classification    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=layers[-1])) 
    ## define code to perform optimization
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)    
    ## add relevant variables to collection before closing the session
    tf.add_to_collection("output_net", layers[-1])
    tf.add_to_collection("loss_net", loss)    
    ## call variable initialization
    init = tf.global_variables_initializer()

    # return network parameters
    return (loss,optimizer,init,x,y)    

def run_training(loss,optimizer,init,x,y,X_training,Y_labels,Training_parameters,path_model,session_datafile,rng):
    '''
    Module running the training   
    '''    
    # Saving the loss  
    costv=[]
    # Starting the saver
    saver = tf.train.Saver()
    # epochs
    epochs = Training_parameters[0]
    # batch size
    batch_size = Training_parameters[1]
    # Starting the session
    with tf.Session() as sess: 
        sess.run(init)
        # training cycle;
        print('\nTraining cycle started...')
        progress = progressbar.ProgressBar()    
        for epoch in progress(range(epochs)):
            avg_cost = 0
            batch_sz = int((X_training.shape[0])/batch_size)
            for i in range(batch_sz):
                # shuffle data at each loop
                shuffle_dts = rng.choice(X_training.shape[0],batch_size)                
                # initializing the input batch
                batch_featurs = X_training[shuffle_dts]
                batch_featurs = np.float32(batch_featurs)
                # initializing the target
                batch_labls = Y_labels[shuffle_dts]    
                batch_labls = np.float32(batch_labls)
                # convert logits for 2 classes instead of 1
                batch_lab_big = np.zeros((batch_labls.shape[0],2))
                batch_lab_big[:,0] = batch_labls.reshape(-1,)
                batch_lab_big[:,1] = np.float32((~batch_labls.astype(bool)).astype(float)).reshape(-1,)
                batch_lab_big = batch_lab_big.reshape(-1,2)                
                # minimizer
                _, BatchLoss = sess.run([optimizer, loss], \
                                        feed_dict={x: batch_featurs, y: batch_lab_big})
                ## compute present epoch loss
                avg_cost += BatchLoss/batch_size
                
            ## print to see where we are
            print("\nEpoch:", (epoch+1), "cost =",\
                  "{:2.5f}".format(avg_cost))              
            costv.append(avg_cost)           
        print('Saving training model and loss.')
        Loss_dic={"loss":np.array(costv)}
        save_to_hdf5(Loss_dic, os.path.join(session_datafile,'training_Loss.h5'))
        saver.save(sess,path_model)   
        print('Training complete!')    

def evaluate_accuracy(path_model,pathgraph,X_validation,X_training,Y_labels,y_test):
    '''
    Module to evaluate the accuracy after training and compare it with logistic regression
    '''
    ### check which prediction went well and which went bad
    # restoring the model
    print('Loading the model..')
    tf.reset_default_graph() 
    with tf.Session() as sess:
        sess = tf.Session()
        meta_file = path_model+'.meta'
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
        # load the parameters for the trained graph
        saver.restore(sess, tf.train.latest_checkpoint(pathgraph))
        # collecting net output
        output_fn = tf.get_collection("output_net")[0]
        # feeding dictionary      
        feed_dict={'input_net:0': X_validation}
        y_compare = np.array(y_test).reshape(-1,1) 
        # argmax 0/1 selects 1st/2nd column
        outbasedonin = 1 - sess.run(tf.argmax(output_fn,1), feed_dict = feed_dict)
        # evaluate the accuracy        
        accuracy = np.mean((outbasedonin.reshape(-1,1)==y_compare).astype(int))
        print('Accuracy on the validation set for the NN model is %4.1f %%'%(100*accuracy))

    ### compare with standard logistic regression
    print(15*'-','\n','Comparing the model with a simple logistic regression in sklearn')    
    logreg = sklr.LogisticRegression(max_iter=200)
    model_LR = logreg.fit(X_training,Y_labels.reshape(-1,))
    # evaluate accuracy
    valp = model_LR.predict(X_validation)
    print('Accuracy on the validation set for the logistic regression model is %4.1f %%'%(100*np.mean((valp.reshape(-1,1)==y_compare).astype(int))))        
    
def main():
    '''    
    This Main module creates the NN model using user-defined parameters and
    runs the training.
    '''
   
    ''' Defining all the parameters for the model 
        We load them from the configuration file
    '''    
    # dataset - path to the positive and negative examples
    ## the training dataset contains 3.6M reviews for training
    dataset_train_path = NLP_dic['dataset_train_path']
    ## the test dataset contains 400K reviews for training
    dataset_test_path = NLP_dic['dataset_train_path']
    # Specify the #of reviews from the training and test dataset to be retained
    N_max_train_test = list(map(int,NLP_dic['N_max_train_test']))

    # defining word representation
    # decide whether you want to use a dense or a 1-hot representation
    # the dense representation is the Google Word2Vec pretrained dictionary
    Use_dense_rep = NLP_dic['Use_dense_rep']
    # If Word2Vec is not used,
    # decide whether you want to count words using tfIdf or simply 1-hot
    tfIdf = NLP_dic['tfIdf']        
    # url to file containing pretrained word2vec representation
    website_pretrained_Word2vec = NLP_dic['website_pretrained_Word2vec']
    # destination folder to download the Word2Vec representation (if needed), or path where to find the file
    Dest_fold_dwl = NLP_dic['Dest_fold_dwl']   

    # defining net and training parameters
    # CPU/GPU platform on which the training should be running
    device_name = NLP_dic['device_name']   
    # path and filename where the training model will be saved
    path_model = NLP_dic['path_model']   
    pathgraph = NLP_dic['pathgraph']   
    
    
    # Define the net geometry: dimension of each layer
    # Dimension of the hidden layer: specify how deep (# of elements in the list) and how
    # wide (value of the elements in the list). If empty, the model will coincide with log-regression
    Dim_hidden = list(map(int,NLP_dic['Dim_hidden']))
    # Define the learning rate and other parameters
    learning_rate = NLP_dic['learning_rate']   
    epochs = int(NLP_dic['epochs'])
    batch_size = int(NLP_dic['batch_size'])
    
    
    ''' Loading the dataset and cleaning the noise '''
    # Loading the dataset and assigning labels        
    print('*Loading the dataset and cleaning the noise*')
    sent_train,sent_test,y_train,y_test = Loading_dataset(dataset_train_path,dataset_test_path,N_max_train_test)
    
    
    # Text cleaning: noise removal
    cleaning_doc(sent_train)
    cleaning_doc(sent_test)
    
    
    ''' Creating a representation for the training vectors and the target labels '''
    # Fullname of the Google Word2Vec dataset
    print('*Creating a vectorial representation for the reviews used for training*')    
    fullfilename = os.path.join(Dest_fold_dwl, 'GoogleNews-vectors-negative300.bin.gz')

    # Construct a vector representation for each review
    X_training,X_validation = Word_representation(sent_train,sent_test,Use_dense_rep,tfIdf, \
                                              website_pretrained_Word2vec,fullfilename,pathgraph)    
    
    
    ''' Creating the neural net model '''
    print('*Creating the neural net model*')
    # random seed and function to shuffle dataset at each batch loop
    seed2 = 228
    rng = np.random.RandomState(seed2)
    Training_parameters = [epochs,batch_size]
    # Define the net geometry
    Dim_net = [X_training.shape[1]]+Dim_hidden+[2]   
    print('*The NN has the following width at each layer*','\n')
    print(Dim_net)
    
    # Creating the NN model
    print('\n1/2 - Creating the nn model.')    
    with tf.device(device_name):
        loss,optimizer,init,x,y = neural_net_model(Dim_net,learning_rate)                    
    print('Done!\n')  
    
    
    # Running the training
    Y_labels = np.array(y_train).reshape(-1,1)
    print('2/2 - Executing the training\n')
    print('Epochs: %4d Batch size: %4d Learn rate: %6.4f Dataset size: %5d \n' \
          %(epochs,batch_size,learning_rate,X_training.shape[0]))
    run_training(loss,optimizer,init,x,y,X_training,Y_labels,Training_parameters,path_model,pathgraph,rng)
    print('Done!\n')      


    ''' Printing out the model accuracy '''   
    print(15*'-','\n','*Evaluating the model accuracy*')     
    evaluate_accuracy(path_model,pathgraph,X_validation,X_training,Y_labels,y_test)
        
if __name__ == "__main__":
    '''    
    Module for NLP sentiment analysis. Training part.  
    Addresses the 'Amazon Reviews for Sentiment Analysis' problem, see:
        https://www.kaggle.com/bittlingmayer/amazonreviews
    '''
    main()    
