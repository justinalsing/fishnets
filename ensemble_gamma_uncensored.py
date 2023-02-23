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
#config_path = 'configs.json'
#with open(config_path) as f:
#        configs = json.load(f)


# where to save the models ?
outdir = "/data80/makinen/fishnets/gamma_pop/results/" #configs["model_outdir"]
outdir = outdir + "model_new_" + sys.argv[1] + "/" # directory labelled by model number
# create output directory
if not os.path.exists(outdir): 
        os.mkdir(outdir)

print('saving models to ', outdir)

# make a plot directory
plotdir = "/home/makinen/repositories/fishnets/gamma_pop/plots/"

print("loading data")


###### load all the data
datadir = "/data80/makinen/fishnets/gamma_pop/data/" #configs["datapath"]

# data sizes
n_sims = 100000
n_data = 500
n_theta = 2

n_test=10000

tmax=10. # days
serum_max_val=4.0

# fiducial parameters for mean, scale
theta_fid = tf.constant([5.0, 0.8], dtype=tf.float32) 
theta_fid_ = theta_fid.numpy()

# prior mean and covariance
priorCinv = tf.eye(n_theta, dtype=tf.float32) 
priormu = tf.zeros(n_theta, dtype=tf.float32)


# load theta and data
theta = np.load(datadir + 'theta_gamma_uncensored.npy')
data = np.load(datadir + 'data_gamma_uncensored.npy')

# make data neural-net friendly
datamax = 500.
tmax = 10.

full_data = np.zeros((n_sims,n_data,n_theta))
full_data[:, :, 0] = data[:, :, 0] / tmax
full_data[:, :, 1] = data[:, :, 1] / datamax
data = full_data; del full_data


# stack up the data and parameters
data = tf.convert_to_tensor(data, dtype=tf.float32)
theta = tf.convert_to_tensor(theta[:, :n_theta], dtype=tf.float32)

# construct masks
score_mask = np.ones((n_sims, n_data, n_theta))
fisher_mask = np.ones((n_sims, n_data, n_theta, n_theta))

score_mask = tf.convert_to_tensor(score_mask, dtype=tf.float32)
fisher_mask = tf.convert_to_tensor(fisher_mask, dtype=tf.float32)


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

        plt.hist(mle[:n_test,0].numpy() - theta[:n_test,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
        plt.xlabel(r'$\hat{\rm mean} - \rm{mean}$')
        plt.legend(frameon=False)
        plt.savefig(checkpoint_folder + 'mean_plot.png')
        plt.close()

        plt.hist(mle[:n_test,1].numpy() - theta[:n_test,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
        plt.xlabel(r'$\hat{\rm scale} - \rm scale$')
        plt.legend(frameon=False)
        plt.savefig(checkpoint_folder + 'scale_plot.png')
        plt.close()

        np.save(checkpoint_folder + 'mle', mle.numpy())
        np.save(checkpoint_folder + 'F', F.numpy() )
        np.save(checkpoint_folder + 'theta', theta.numpy())


# initialize the Fishnet model
Model = FishnetTwin(n_parameters=n_theta, 
                n_inputs=2, 
                n_hidden_score=[256, 256, 256], 
                activation_score=[tf.nn.elu, tf.nn.elu,  tf.nn.elu],
                n_hidden_fisher=[256, 256, 256], 
                activation_fisher=[tf.nn.elu, tf.nn.elu,  tf.nn.elu],
                optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                theta_fid=theta_fid,
                priormu=tf.zeros(n_theta, dtype=tf.float32),
                priorCinv=tf.eye(n_theta, dtype=tf.float32))

# add in corrected loss function
@tf.function
def construct_fisher_matrix(outputs):
    
    Q = tfp.math.fill_triangular(outputs)
    # EDIT: changed to + softplus(diag_part(Q))
    L = Q - tf.linalg.diag(tf.linalg.diag_part(Q) - tf.math.softplus(tf.linalg.diag_part(Q)))
    return tf.einsum('...ij,...jk->...ik', L, tf.transpose(L, perm=[0, 1, 3, 2]))

Model.construct_fisher_matrix = construct_fisher_matrix

print("commencing training")

# train model with batch size 512
Model.train((data, theta, score_mask, fisher_mask), lr=5e-4, epochs=100)
model_checkpoint(Model=Model, checkpoint=0)
Model.train((data, theta, score_mask, fisher_mask), lr=1e-4, epochs=100)
model_checkpoint(Model=Model, checkpoint=1)
Model.train((data, theta, score_mask, fisher_mask), lr=5e-5, epochs=100)
model_checkpoint(Model=Model, checkpoint=2)
Model.train((data, theta, score_mask, fisher_mask), lr=1e-6, epochs=200)
model_checkpoint(Model=Model, checkpoint=3)
Model.train((data, theta, score_mask, fisher_mask), lr=5e-7, epochs=200)
model_checkpoint(Model=Model, checkpoint=4)



# save model before LBFGS in case the optimization fails

# do LBFGS optimization on the full dataset (this might fail)
#Model.lbfgs_optimize(data, theta, score_mask, fisher_mask, max_iterations=10, tolerance=1e-5)

# if the above is sucessful, save model outputs
Model.save(outdir + 'model')

# ----- LOAD BIG TEST DATA -----
n_sims = 5000
n_data = 10000

# more sims; n_sims=5000
data_test = np.load('/data80/makinen/fishnets/gamma_pop/uncensored_targets/training_sims_10k.npy')
#np.load(datadir + 'data_uncensored_10k.npy')
theta_test = np.load('/data80/makinen/fishnets/gamma_pop/uncensored_targets/training_theta_10k.npy')
#np.load(datadir + 'theta_uncensored_10k.npy')
# make data neural-net friendly
datamax = 500.
tmax = 10.

data_test[:, :, 0] /= tmax
data_test[:, :, 1] /= datamax

# stack up the data and parameters
data = tf.convert_to_tensor(data_test, dtype=tf.float32)
theta = tf.convert_to_tensor(theta_test[:, :n_theta], dtype=tf.float32)

# construct masks
score_mask = np.ones((n_sims, n_data, n_theta))
fisher_mask = np.ones((n_sims, n_data, n_theta, n_theta))

score_mask = tf.convert_to_tensor(score_mask, dtype=tf.float32)
fisher_mask = tf.convert_to_tensor(fisher_mask, dtype=tf.float32)

# model MLEs
mle, F = Model.compute_mle_(data, score_mask, fisher_mask)

# save mles
np.save(outdir + "test_mle", mle)
np.save(outdir + "test_F", F)
np.save(outdir + "test_theta", theta.numpy())

plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.xlabel('mle_mean - mean')
plt.legend(frameon=False)
plt.savefig(outdir + 'uncensored_mean_test_%d.png'%(int(sys.argv[1])))
plt.savefig(plotdir + 'uncensored_mean_test_%d.png'%(int(sys.argv[1])))
plt.close()

plt.hist(mle[:,1].numpy() - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.xlabel('mle_scale - scale')
plt.legend(frameon=False)
plt.savefig(outdir + 'uncensored_scale_test_%d.png'%(int(sys.argv[1])))
plt.savefig(plotdir + 'uncensored_scale_test_%d.png'%(int(sys.argv[1])))
plt.close()