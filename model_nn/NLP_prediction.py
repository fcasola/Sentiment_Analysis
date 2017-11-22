"""
@author: Francesco Casola, Harvard University
fr.casola@gmail.com

Prediction module for the NLP analysis of Amazon reviews
"""
import tensorflow as tf
import numpy as np, h5py, os
from collections import Counter
# personal modules
import NLP_training
from NLP_config import *

def load_from_hdf5(filename):
    """
    Compact way to load hierarchical data format, Part I
    
        Input parameters are
        *filename: full *.h5 name of the file to load 
    """
    with h5py.File(filename, 'r') as h5file:
        return load_dict_contents(h5file, '/')

def load_dict_contents(h5file, path):
    """
    Compact way to load hierarchical data format, Part II
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
    return ans

def check_voc_avail(filename):
    '''
    Module checking the existance and loading the restricted vectgorized vocabulary
    '''
    if os.path.exists(filename):
        print('Loading dictionary. Pls. Wait..')
        dic_file = load_from_hdf5(filename)
        print('Done!')
        return True,dic_file
    else:    
        return False,None
    
def convert_review(review,Vector_Vocab):
    '''
    Convert the review from text to vector
    
    convert_review(review,Vector_Vocab)
        -review: review to be converted
        -Vector_Vocab: reduced vocabulary to be used
    '''    
    main_text_rev = [review]
    # cleaning the text
    NLP_training.cleaning_doc(main_text_rev)
    # construct vector
    counts_mis = Counter()
    counts_tot = Counter()   
    # Initialize the training vector
    vec_size = max(Vector_Vocab[list(Vector_Vocab.keys())[0]].shape)
    X_training = np.zeros((1,vec_size))
    for word in main_text_rev[0].split(): 
        counts_tot[word]+=1
        if word in Vector_Vocab:
            X_training[0,:] = np.add(X_training[0,:], Vector_Vocab[word].reshape(1,-1))  
        else:
            counts_mis[word]+=1
    return X_training
    print('Training vecor created. %d words out of %d were not in the training dictionary'%(len(counts_mis),len(counts_tot)))        

def interactive_console(meta_file,pathgraph,Vector_Vocab):
    '''
    Module running predictions
    '''    
    tf.reset_default_graph() 
    with tf.Session() as sess:
        sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
        # load the parameters for the trained graph
        saver.restore(sess, tf.train.latest_checkpoint(pathgraph))
        # collecting net output
        output_fn = tf.get_collection("output_net")[0]
        # starting the interactive console
        review = ''
        print(15*'-')
        while review != 'exit()':
            review = input('Please enter your review (type exit() to quit): ')
            # put user's sentence into a vector
            X_training = convert_review(review,Vector_Vocab)
            # feeding dictionary      
            feed_dict={'input_net:0': X_training}
            # argmax 0/1 selects 1st/2nd column
            outbasedonin = 1 - sess.run(tf.argmax(output_fn,1), feed_dict = feed_dict)
            if outbasedonin == 1:
                print('Nice! if this is what you think, the product deserves 4/5 stars!')
            else:
                print('Oh I am sorry! if this is what you think, the product deserves 1/2 stars!')            
            
def main():
    '''
    This main module loads the trained model and an interactive console 
    where the use can type his new review and retrieve the predicted product 
    amazon stars
    '''
    
    '''Define parameters for the prediction'''
    # path and filename where the training model will be saved
    path_model = NLP_dic['path_model']   
    pathgraph = NLP_dic['pathgraph'] 
    file_vocabulary = os.path.join(pathgraph,'dictionary_model.h5')
    file_model = path_model+'.meta'
    
    ''' Loading model files and starting the interactive console '''
    print('*Module running interactive NLP predictions*')
    # loading the restricted vectorized vocabulary
    Isthere,Vector_Vocab = check_voc_avail(file_vocabulary)
    
    if Isthere:
        # First parts, load the tensorflow model.
        if os.path.exists(file_model):
            interactive_console(file_model,pathgraph,Vector_Vocab)
        else:
            raise Exception('Missing tensorflow model; prediction not implemented at the moment.')             
    else:
        raise Exception('Missing vocabulary; prediction not implemented at the moment.') 
    
if __name__ == "__main__":
    '''    
    Module for NLP sentiment analysis. Prediction part.  
    Addresses the 'Amazon Reviews for Sentiment Analysis' problem, see:
        https://www.kaggle.com/bittlingmayer/amazonreviews
    '''
    main() 


