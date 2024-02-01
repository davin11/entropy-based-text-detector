# Testing CODE
import os
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.svm import OneClassSVM
from transformers import CodeGenTokenizer, AutoModelForCausalLM
from joblib import dump, load
from train import feature_extraction

if __name__=="__main__":
    #set test file
    from sys import argv
    test_csv = './test_essays.csv' if len(argv)<2 else argv[1]
    
    #set DEVICE and DTYPE
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(device, dtype, flush=True)

    # MAKE LLM
    llm_tokenizer = CodeGenTokenizer.from_pretrained("microsoft/phi-2",
                                                     add_bos_token = True, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=dtype,
                                                     device_map=device, trust_remote_code=True)
    max_length = 2048
    
    # laod test file
    test_tab = pd.read_csv(test_csv)
    
    # feature_extraction
    batch_size = 3
    test_tab, feats_list = feature_extraction(test_tab, batch_size, llm_model, llm_tokenizer, max_length)
    test_feats = test_tab[feats_list].values

    # zscore
    z_mean = np.load('zscore.npz')['z_mean']
    z_std  = np.load('zscore.npz')['z_std']
    test_feats = (test_feats - z_mean)/z_std
    
    # load classifier
    classifier = load('oneClassSVM.joblib')
    
    # predict
    test_tab['generated'] = -1.0*classifier.decision_function(test_feats)
    
    # Save the DataFrame to a CSV file
    test_tab[['id','generated']].to_csv('submission.csv', index=False)
    
