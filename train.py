# Training CODE
import os
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.svm import OneClassSVM
from transformers import CodeGenTokenizer, AutoModelForCausalLM
from joblib import dump, load

def compute_entropy(input_ids, logits, attention_mask, token_type_ids=None):
    # compute information given the tokens and logits
    # it returns:
    #  entD : expected information for each token
    #  entL : given information for each token
    with torch.no_grad():
        logits = torch.log_softmax(logits.float(), dim=-1)
        
        tokens = input_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]
        
        entD = torch.sum(logits * torch.exp(logits), dim=-1)[:, 1:]
        entL = torch.gather(logits[:, :-1, :], dim=-1, index = tokens[:,:,None])[:,:,0]
        
        entD = -torch.where(attention_mask!=0, entD, np.nan)
        entL = -torch.where(attention_mask!=0, entL, np.nan)
        
    return entD, entL

class Batch:
    # batch a list
    def __init__(self, iterable, size=1):
        self.iterable = iterable
        self.size = size
        self.len = len(range(0, len(self.iterable), self.size))
        
    def __iter__(self):
        l = len(self.iterable)
        n = self.size
        for ndx in range(0, l, n):
            yield self.iterable[ndx:min(ndx + n, l)]
    
    def __len__(self):
        return self.len

def feature_extraction(tab, batch_size, llm_model, llm_tokenizer, max_length):
    # feature extraction
    #
    for index in tqdm.tqdm(tab.index):
        text = tab.loc[index,'text']
        tab.loc[index,'len_chr'] = len(text)
    
    feats_list = ['Dmed', 'Lmed', 'Dp05', 'Lstd', 'meanchr',]
    with torch.no_grad():
        device = next(llm_model.parameters()).device
        for index_list in tqdm.tqdm(Batch(tab.index, batch_size)):
            texts = [tab.loc[index,'text'] for index in index_list]
            
            # run LLM
            tokens = llm_tokenizer(texts, return_tensors="pt",
                                   max_length=max_length, truncation=max_length is not None,
                                   truncation_strategy = 'longest_first',
                                   add_special_tokens=True, padding=True)
            tokens = {_: tokens[_].to(device) for _ in tokens}
            logits = llm_model(**tokens).logits
            
            # compute entropy
            vetD, vetL = compute_entropy(logits=logits, **tokens)
            vetD = vetD.cpu().numpy()
            vetL = vetL.cpu().numpy()

            # compute features
            tab.loc[index_list,'meanchr'] = tab.loc[index_list,'len_chr'].values / np.sum(np.isfinite(vetL),-1)
            tab.loc[index_list, 'Dmed'] = np.nanmedian(vetD, axis=-1)
            tab.loc[index_list, 'Lmed'] = np.nanmedian(vetL, axis=-1)
            tab.loc[index_list, 'Dp05'] = np.nanpercentile(vetD, 5, axis=-1)
            tab.loc[index_list, 'Lstd'] = np.nanstd(vetL, axis=-1)
            
    return tab, feats_list

if __name__=="__main__":
    #set training file
    from sys import argv
    train_csv = './train_essays.csv' if len(argv)<2 else argv[1]
    
    #set DEVICE and DTYPE
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(device, dtype, flush=True)

    # MAKE LLM
    llm_tokenizer = CodeGenTokenizer.from_pretrained("microsoft/phi-2", add_bos_token = True, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=dtype,
                                                     device_map=device, trust_remote_code=True)
    max_length = 2048
    
    # laod training file
    train_tab = pd.read_csv(train_csv)
    
    # feature_extraction
    batch_size = 3
    train_tab, feats_list = feature_extraction(train_tab, batch_size, llm_model, llm_tokenizer, max_length)
    
    # take only feature of real data
    train_feats = train_tab[train_tab['generated']==0][feats_list].values
    
    # zscore
    z_mean = np.mean(train_feats, 0, keepdims=True)
    z_std  = np.maximum(np.std(train_feats, 0, keepdims=True), 1e-4)
    train_feats = (train_feats - z_mean)/z_std
    np.savez('zscore.npz', z_std=z_std, z_mean=z_mean)
    
    # train classifier
    classifier = OneClassSVM(verbose=1,  kernel='rbf', gamma='auto',nu=0.05)
    classifier.fit(train_feats)
    dump(classifier, 'oneClassSVM.joblib')
