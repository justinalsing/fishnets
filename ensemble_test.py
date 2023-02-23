import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import tensorflow_probability as tfp
from tqdm import trange
from scipy import stats
tfk = tf.keras
import json
import sys, os

from fishnets import *


##### configs #####
config_path = 'configs.json'
with open(config_path) as f:
        configs = json.load(f)

# where to save the models ?
outdir = "/data80/makinen/fishnets/results/ensemble_training/" #configs["model_outdir"]
outdir = outdir + "model_" + sys.argv[1] + "/" # directory labelled by model number
# create output directory
if not os.path.exists(outdir): 
        os.mkdir(outdir)

print('saving models to ', outdir)


print("loading data")

# data sizes
n_sims = 10000
n_data = 10000

# fiducial parameters
theta_fid = tf.constant([0.,0.], dtype=tf.float32)
theta_fid_ = theta_fid.numpy()

# prior mean and covariance
priorCinv = tf.convert_to_tensor(np.eye(2), dtype=tf.float32)
priormu = tf.constant([0.,0.], dtype=tf.float32)


###### load all the data
datadir = "/data80/makinen/fishnets/" #configs["datapath"]


data  = tf.convert_to_tensor(np.load(datadir + "ensemble_test/data.npy"), dtype=tf.float32)
theta = tf.convert_to_tensor(np.load(datadir + "ensemble_test/theta.npy"), dtype=tf.float32)
score_mask = tf.convert_to_tensor(np.load(datadir + "ensemble_test/score_mask.npy"), dtype=tf.float32)
fisher_mask = tf.convert_to_tensor(np.load(datadir + "ensemble_test/fisher_mask.npy"), dtype=tf.float32)

F_ = np.load(datadir + "ensemble_test/F_.npy")
t_ = np.load(datadir + "ensemble_test/t_.npy")
pmle_ = np.load(datadir + "ensemble_test/pmle_.npy")



print("loading model")
model_filename = outdir + 'model'

# initialize the Fishnet model
Model = FishnetTwin(n_parameters=2, 
                n_inputs=5, 
                n_hidden_score=[256, 256, 256], 
                activation_score=[tf.nn.elu, tf.nn.elu, tf.nn.elu],
                n_hidden_fisher=[256, 256, 256], 
                activation_fisher=[tf.nn.elu, tf.nn.elu, tf.nn.elu],
                optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                theta_fid=theta_fid,
                priormu=tf.zeros(2, dtype=tf.float32),
                priorCinv=tf.eye(2, dtype=tf.float32), 
                restore=True, restore_filename=model_filename)



# make plots on test data
print("computing F and mle with model on test data")

# model MLEs
mle, F = Model.compute_mle_(data, score_mask, fisher_mask)

# save mles
np.save(outdir + "pred_mle_test", mle)
np.save(outdir + "pred_F_test", F)

plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.hist(pmle_[:,0] - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,0] - theta[:,0].numpy())
x = np.linspace(-4*std, 4*std, 500)
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
#plt.axvline(np.mean(mle[:,0].numpy() - theta[:,0].numpy()))
#plt.axvline(np.mean(pmle_[:,0] - theta[:,0].numpy()))
plt.xlabel('$\hat{m} - m$')
plt.legend(frameon=False)
plt.savefig(outdir + 'm_plot_test.png')
plt.close()

plt.hist(mle[:,1].numpy() - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.hist(pmle_[:,1] - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,1] - theta[:,1].numpy())
x = np.linspace(-4*std, 4*std, 500)
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
#plt.axvline(np.mean(mle[:,1].numpy() - theta[:,1].numpy()), color='blue')
#plt.axvline(np.mean(pmle_[:,1] - theta[:,1].numpy()), color='orange')
plt.xlabel('$\hat{c} - c$')
plt.legend(frameon=False)
plt.savefig(outdir + 'c_plot_test.png')
plt.close()