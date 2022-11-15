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

print("creating data for ensemble training now ...")


# data sizes
n_sims = 10000 #configs["n_sims"]
n_data = 10000 #configs["n_data_train"]

masked = False

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

# put in some "inductive biases"
x1_ = x_*(1./ sigma_**2)
x2_ = y_*(1./ sigma_**2)

# stack up the data and parameters
data = tf.stack([y_, x_, 1./sigma_**2, x1_, x2_], axis=-1)
theta = tf.stack([m_, c_], axis=-1)


# construct masks
score_mask = np.ones((n_sims, n_data, 2))
fisher_mask = np.ones((n_sims, n_data, 2, 2))

#     # make the masks
# if masked:
#     for i in range(n_sims):

#         # how many points to mask?
#         n_mask = np.random.randint(1, n_data-5)

#         # choose which points to mask
#         idx = np.random.choice(np.arange(n_data), n_mask, replace=False)

#         # mask those points (set the fisher and score masks to zero for those points)
#         for j in idx:
#             score_mask[i,j,:] = 0
#             fisher_mask[i,j,...] = 0
# else:
#     pass


# compute MLEs
F_ = np.sum(np.stack([x_**2 / sigma_**2, x_ / sigma_**2, x_ / sigma_**2, 1. / sigma_**2], axis=-1).reshape((n_sims, n_data, 2, 2)) \
         * fisher_mask, axis=1) + priorCinv.numpy()
t_ = np.sum(np.stack([x_*(y_ - (theta_fid[0]*x_ + theta_fid[1]))/ sigma_**2, (y_ - (theta_fid[0]*x_ + theta_fid[1])) / sigma_**2], axis=-1) \
        * score_mask, axis=1) - np.dot(priorCinv, theta_fid - priormu)
pmle_ = theta_fid_ + np.einsum('ijk,ik->ij', np.linalg.inv(F_), t_)




###### save all the data
outdir = configs["datapath"]

print("saving data to ", outdir)

np.save(outdir + "ensemble_test/data", data.numpy())
np.save(outdir + "ensemble_test/theta", theta.numpy())
np.save(outdir + "ensemble_test/score_mask", score_mask)
np.save(outdir + "ensemble_test/fisher_mask", fisher_mask)
np.save(outdir + "ensemble_test/F_", F_)
np.save(outdir + "ensemble_test/t_", t_)
np.save(outdir + "ensemble_test/pmle_", pmle_)