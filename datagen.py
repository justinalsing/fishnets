import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
from scipy import stats
tfk = tf.keras
import json


config_path = 'configs.json'
with open(config_path) as f:
        configs = json.load(f)




# data sizes
n_sims = configs["n_sims"]
n_data = configs["n_data_train"]

# fiducial parameters
theta_fid = tf.constant([0.,0.], dtype=tf.float32)
theta_fid_ = theta_fid.numpy()

# prior mean and covariance
priorCinv = tf.convert_to_tensor(np.eye(2), dtype=tf.float32)
priormu = tf.constant([0.,0.], dtype=tf.float32)

# slopes and intercepts
m_ = np.random.normal(0, 1, n_sims).astype(np.float32)
c_ = np.random.normal(0, 1, n_sims).astype(np.float32)

# x-values
x_ = np.random.uniform(0, 10, (n_sims, n_data)).astype(np.float32)

# noise std devs
sigma_ = np.random.uniform(1, 10, (n_sims, n_data)).astype(np.float32)

# simulate "data"
y_ = m_[...,np.newaxis]*x_ + c_[...,np.newaxis] + np.random.normal(0, 1, sigma_.shape)*sigma_
y_ = y_.astype(np.float32)

# stack up the data and parameters
data = tf.stack([y_, x_, 1./sigma_**2], axis=-1)
theta = tf.stack([m_, c_], axis=-1)


# construct masks
score_mask = np.ones((n_sims, n_data, 2))
fisher_mask = np.ones((n_sims, n_data, 2, 2)

# make the masks
if masked is True:
    for i in range(n_sims):
        
        # how many points to mask?
        n_mask = np.random.randint(1, n_data-5)
        
        # choose which points to mask
        idx = np.random.choice(np.arange(n_data), n_mask, replace=False)
        
        # mask those points (set the fisher and score masks to zero for those points)
        for j in idx:
            score_mask[i,j,:] = 0
            fisher_mask[i,j,...] = 0




###### save all the data
outdir = configs["datapath"]



np.save(outdir + "data", data.numpy())
np.save(outdir + "theta", theta.numpy())
np.save(outdir + "score_mask", score_mask)
np.save(outdir + "fisher_mask", fisher_mask)