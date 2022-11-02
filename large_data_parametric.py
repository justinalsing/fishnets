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
outdir = configs["model_outdir"]

outdir += sys.argv[1] + "_"


# data sizes
n_sims = 10000
n_data = 10000

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
sigma_ = np.random.uniform(1, 5, (n_sims, n_data)).astype(np.float32)

# simulate "data"
y_ = m_[...,np.newaxis]*x_ + c_[...,np.newaxis] + np.random.normal(0, 1, sigma_.shape)*sigma_
y_ = y_.astype(np.float32)


# put in some "inductive biases"
x1_ = x_*(1./ sigma_**2)
x2_ = y_*(1./ sigma_**2)
#x3_ = x_*(sigma_**2)

# stack up the data and parameters
data = tf.stack([y_, x_, 1./sigma_**2, x1_, x2_], axis=-1)
theta = tf.stack([m_, c_], axis=-1)

# construct masks
score_mask = np.ones((n_sims, n_data, 2))
fisher_mask = np.ones((n_sims, n_data, 2, 2))

score_mask = tf.convert_to_tensor(score_mask, dtype=tf.float32)
fisher_mask = tf.convert_to_tensor(fisher_mask, dtype=tf.float32)


# compute MLEs
F_ = np.sum(np.stack([x_**2 / sigma_**2, x_ / sigma_**2, x_ / sigma_**2, 1. / sigma_**2], axis=-1).reshape((n_sims, n_data, 2, 2)) * fisher_mask.numpy(), axis=1) + priorCinv.numpy()
t_ = np.sum(np.stack([x_*(y_ - (theta_fid[0]*x_ + theta_fid[1]))/ sigma_**2, (y_ - (theta_fid[0]*x_ + theta_fid[1])) / sigma_**2], axis=-1) * score_mask.numpy(), axis=1) - np.dot(priorCinv, theta_fid - priormu)
pmle_ = theta_fid_ + np.einsum('ijk,ik->ij', np.linalg.inv(F_), t_)



# define checkpoint function
def model_checkpoint(Model, checkpoint=0):
        print("reached checkpoint %d, saving model"%(checkpoint))
        checkpoint_folder = outdir + 'checkpoint_%d/'%(checkpoint)
        os.mkdir(checkpoint_folder)

        Model.save(checkpoint_folder + 'model')

        # model MLEs
        mle, F = Model.compute_mle_(data, score_mask, fisher_mask)

        plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
        plt.hist(pmle_[:,0] - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
        std = np.std(pmle_[:,0] - theta[:,0].numpy())
        x = np.linspace(-4*std, 4*std, 500)
        plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
        #plt.axvline(np.mean(mle[:,0].numpy() - theta[:,0].numpy()))
        #plt.axvline(np.mean(pmle_[:,0] - theta[:,0].numpy()))
        plt.xlabel('$\hat{m} - m$')
        plt.legend(frameon=False)
        plt.savefig(checkpoint_folder + 'm_plot.png')
        plt.close()

        plt.hist(mle[:,1].numpy() - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
        plt.hist(pmle_[:,1] - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
        std = np.std(pmle_[:,1] - theta[:,1].numpy())
        x = np.linspace(-4*std, 4*std, 500)
        plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
        #plt.axvline(np.mean(mle[:,1].numpy() - theta[:,1].numpy()), color='blue')
        #plt.axvline(np.mean(pmle_[:,1] - theta[:,1].numpy()), color='orange')
        plt.xlabel('$\hat{c} - c$')
        plt.legend(frameon=False)
        plt.savefig(checkpoint_folder + 'c_plot.png')
        plt.close()
        


# initialize the Fishnet model
Model = FishnetTwinParametric(n_parameters=2, 
                n_inputs=5, 
                n_hidden_score=[128, 128, 128], 
                #activation_score=[tf.nn.elu, tf.nn.elu, tf.nn.elu],
                n_hidden_fisher=[128, 128, 128], 
                #activation_fisher=[tf.nn.elu, tf.nn.elu, tf.nn.elu]
                optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                theta_fid=theta_fid,
                priormu=tf.zeros(2, dtype=tf.float32),
                priorCinv=tf.eye(2, dtype=tf.float32))

# train model
Model.train((data, theta, score_mask, fisher_mask), lr=5e-4, epochs=500)
model_checkpoint(Model=Model, checkpoint=0)
Model.train((data, theta, score_mask, fisher_mask), lr=1e-4, epochs=500)
model_checkpoint(Model=Model, checkpoint=1)
Model.train((data, theta, score_mask, fisher_mask), lr=5e-5, epochs=500)
model_checkpoint(Model=Model, checkpoint=2)
Model.train((data, theta, score_mask, fisher_mask), lr=1e-6, epochs=250)
model_checkpoint(Model=Model, checkpoint=3)
Model.train((data, theta, score_mask, fisher_mask), lr=5e-6, epochs=250)
model_checkpoint(Model=Model, checkpoint=4)


# save model before LBFGS in case the optimization fails

# do LBFGS optimization on the full dataset (this might fail)

#Model.lbfgs_optimize(data, theta, score_mask, fisher_mask, max_iterations=10, tolerance=1e-5)




# if the above is sucessful, save model outputs
Model.save(outdir + 'model')

# make plots

# model MLEs
mle, F = Model.compute_mle_(data, score_mask, fisher_mask)

plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.hist(pmle_[:,0] - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,0] - theta[:,0].numpy())
x = np.linspace(-4*std, 4*std, 500)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
#plt.axvline(np.mean(mle[:,0].numpy() - theta[:,0].numpy()))
#plt.axvline(np.mean(pmle_[:,0] - theta[:,0].numpy()))
plt.xlabel('$\hat{m} - m$')
plt.legend(frameon=False)
plt.savefig(outdir + 'm_plot.png')
plt.close()

plt.hist(mle[:,1].numpy() - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.hist(pmle_[:,1] - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,1] - theta[:,1].numpy())
x = np.linspace(-4*std, 4*std, 500)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
#plt.axvline(np.mean(mle[:,1].numpy() - theta[:,1].numpy()), color='blue')
#plt.axvline(np.mean(pmle_[:,1] - theta[:,1].numpy()), color='orange')
plt.xlabel('$\hat{c} - c$')
plt.legend(frameon=False)
plt.savefig(outdir + 'c_plot.png')
plt.close()