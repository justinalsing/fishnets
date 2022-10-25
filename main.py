import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import tensorflow_probability as tfp
from tqdm import trange
from scipy import stats
tfk = tf.keras
import json

from fishnets import *


##### configs #####
config_path = 'configs.json'
with open(config_path) as f:
        configs = json.load(f)

# where to save the models ?
outdir = ""


# data sizes
n_sims = configs["n_sims"]
n_data = configs["n_data_train"]

# fiducial parameters
theta_fid = tf.constant([0.,0.], dtype=tf.float32)
theta_fid_ = theta_fid.numpy()



##### load data #####
datapath = configs["datapath"]

data = np.load(datapath + "data.npy")
theta = np.load(datapath + "theta.npy")

# unpack data
x_ = data[:, :, 1]
y_ = data[:, :, 0]
sigma_ = np.sqrt(1. / data[:, :, 2])

# unpack params
m_ = theta[:, 0]
c_ = theta[:, 1]


data = tf.convert_to_tensor(data, dtype=tf.float32)
theta = tf.convert_to_tensor(theta, dtype=tf.float32)

score_mask = np.load(datapath + "score_mask.npy")
fisher_mask = np.load(datapath + "fisher_mask.npy")
score_mask = tf.convert_to_tensor(score_mask, dtype=tf.float32)
fisher_mask = tf.convert_to_tensor(fisher_mask, dtype=tf.float32)





# compute MLEs
F_ = np.sum(np.stack([x_**2 / sigma_**2, x_ / sigma_**2, x_ / sigma_**2, 1. / sigma_**2], axis=-1).reshape((n_sims, n_data, 2, 2)) * fisher_mask.numpy(), axis=1) + priorCinv.numpy()
t_ = np.sum(np.stack([x_*(y_ - (theta_fid[0]*x_ + theta_fid[1]))/ sigma_**2, (y_ - (theta_fid[0]*x_ + theta_fid[1])) / sigma_**2], axis=-1) * score_mask.numpy(), axis=1) - np.dot(priorCinv, theta_fid - priormu)
pmle_ = theta_fid_ + np.einsum('ijk,ik->ij', np.linalg.inv(F_), t_)


# initialize the Fishnet model
Model = FishnetTwin(n_parameters=2, 
                n_inputs=3, 
                n_hidden_score=[64, 64], 
                activation_score=[tf.nn.elu, tf.nn.elu],
                n_hidden_fisher=[64, 64], 
                activation_fisher=[tf.nn.elu, tf.nn.elu],
                optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                theta_fid=theta_fid,
                priormu=tf.zeros(2, dtype=tf.float32),
                priorCinv=tf.eye(2, dtype=tf.float32))

# save model before LBFGS in case the optimization fails


# do LBFGS optimization on the full dataset (this might fail)

Model.lbfgs_optimize(data, theta, score_mask, fisher_mask, max_iterations=10, tolerance=1e-5)

# if the above is sucessful, save model outputs
Model.save(outdir)