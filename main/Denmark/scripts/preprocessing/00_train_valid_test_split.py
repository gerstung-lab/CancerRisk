# %% [markdown]
# # Train Validation Test split for DB data
# 
# - script to produce the corresponding split for the files in DB/DB/raw
# - Test - 25%   
# - Valid - 5%   
# - Train - 65%  
# 
# ## Output: 
# 
# - file numbers fro each split
# 
# 'DB/DB/raw/trainvalidtest.pickle'
# 

# %%
import sys 
import os 
import pickle
import numpy as np 
np.random.seed(7)

# %%
d_data = '/home/people/alexwolf/data/'

# %%
idx = np.arange(1000)

# %%
test = np.random.choice(idx, 250, replace=False)
idx = np.setdiff1d(idx, test)
valid = np.random.choice(idx, 50, replace=False)
train = np.setdiff1d(idx, valid)


# %%
print((np.sum(train) + np.sum(valid) + np.sum(test)) == 499500)
print((len(train) + len(valid) + len(test)) == 1000)

# %%
with open(d_data + 'DB/DB/raw/trainvalidtest.pickle', 'wb') as handle:
    pickle.dump({'train': train.tolist(), 
                 'valid': valid.tolist(), 
                 'test': test.tolist()}, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%



