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
n_data = 500

# fiducial parameters
theta_fid = tf.constant([0.,0.], dtype=tf.float32)
theta_fid_ = theta_fid.numpy()

# prior mean and covariance
priorCinv = tf.convert_to_tensor(np.eye(2), dtype=tf.float32)
priormu = tf.constant([0.,0.], dtype=tf.float32)


###### load all the data
datadir = "/data80/makinen/fishnets/" #configs["datapath"]


data  = tf.convert_to_tensor(np.load(datadir + "ensemble/data.npy"), dtype=tf.float32)
theta = tf.convert_to_tensor(np.load(datadir + "ensemble/theta.npy"), dtype=tf.float32)
score_mask = tf.convert_to_tensor(np.load(datadir + "ensemble/score_mask.npy"), dtype=tf.float32)
fisher_mask = tf.convert_to_tensor(np.load(datadir + "ensemble/fisher_mask.npy"), dtype=tf.float32)

F_ = np.load(datadir + "ensemble/F_.npy")
t_ = np.load(datadir + "ensemble/t_.npy")
pmle_ = np.load(datadir + "ensemble/pmle_.npy")




# define checkpoint function
def model_checkpoint(Model, checkpoint=0):
        print("reached checkpoint %d, saving model"%(checkpoint))
        checkpoint_folder = outdir + 'checkpoint_%d/'%(checkpoint)
            # create output directory
        if not os.path.exists(checkpoint_folder): 
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
Model = FishnetTwin(n_parameters=2, 
                n_inputs=5, 
                n_hidden_score=[256, 256, 256], 
                activation_score=[tf.nn.elu, tf.nn.elu, tf.nn.elu],
                n_hidden_fisher=[256, 256, 256], 
                activation_fisher=[tf.nn.elu, tf.nn.elu, tf.nn.elu],
                optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                theta_fid=theta_fid,
                priormu=tf.zeros(2, dtype=tf.float32),
                priorCinv=tf.eye(2, dtype=tf.float32))

print("commencing training")

# train model with batch size 512
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

# save mles
np.save(outdir + "pred_mle", mle)
np.save(outdir + "pred_F", F)

plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.hist(pmle_[:,0] - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,0] - theta[:,0].numpy())
x = np.linspace(-4*std, 4*std, 500)
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
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
#plt.plot(x, stats.norm.pdf(x, loc=0, scale=std), color='orange')
#plt.axvline(np.mean(mle[:,1].numpy() - theta[:,1].numpy()), color='blue')
#plt.axvline(np.mean(pmle_[:,1] - theta[:,1].numpy()), color='orange')
plt.xlabel('$\hat{c} - c$')
plt.legend(frameon=False)
plt.savefig(outdir + 'c_plot.png')
plt.close()