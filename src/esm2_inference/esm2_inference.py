import torch
import pandas as pd
import esm
import os
import numpy as np

# SETUP DEVICE 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")  # This should print 'cuda'

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device) # Move model to GPU
batch_converter = alphabet.get_batch_converter()
model.eval()

BATCH_SIZE = 50 

os.makedirs("./results", exist_ok=True) # results your output folder

df = pd.read_csv("./main_inputs.tsv", sep="\t") # your main_input id_transcrip\tsequence\nseqid1\tMPRFSATTKLRTFAGMQIPHSSTKAVQGSEHGVYFHWLGKWRFTVIRGF\n etc ...
data = list(zip(df["id_transcript"], df["sequence"])) # note this header columns

# Optimize: Sort data by length. This minimizes padding and speeds up inference significantly.
data.sort(key=lambda x: len(x[1]), reverse=True) 

for i in range(0, len(data), BATCH_SIZE):
    batch_data = data[i:i+BATCH_SIZE]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    
    # 2. MOVE INPUTS TO GPU (CRITICAL FIX)
    batch_tokens = batch_tokens.to(device)
    
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        # Only extract the specific layer you need to save memory
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]

    for j, tokens_len in enumerate(batch_lens):
        # 1:tokens_len-1 removes the start (<cls>) and end (<eos>) tokens
        seq_embedding = token_representations[j, 1:tokens_len-1].mean(0)
        
        # Move back to CPU for saving
        embedding_array = seq_embedding.cpu().numpy() 
        
        # Save
        np.save(f"../results2_bacteria/{batch_labels[j]}_esm2embedding.npy", embedding_array)

    if i % 1000 == 0:
        print(f"Processed {i} / {len(data)} sequences")
