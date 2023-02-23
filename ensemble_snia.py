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
outdir = "/data80/makinen/fishnets/snia_results/" #configs["model_outdir"]
outdir = outdir + "model_full_" + sys.argv[1] + "/" # directory labelled by model number
# create output directory
if not os.path.exists(outdir): 
        os.mkdir(outdir)

print('saving models to ', outdir)


print("loading data")


###### load all the data
datadir = "/data80/makinen/fishnets/snia/" #configs["datapath"]

# data sizes
n_sims = 100000 #int(1e6)
n_data = 500
n_theta = 10

# fiducial parameters for Om, w
#theta_fid = tf.constant([0.3, -1.0], dtype=tf.float32)
theta_fid = tf.constant([0.3, -1.0,  0.13, 2.56, -19.3, 0.1, 0.0, 1.0, 0.0, 0.1], dtype=tf.float32)
theta_fid_ = theta_fid[:n_theta].numpy()

# prior mean and covariance
priorCinv = tf.eye(n_theta, dtype=tf.float32) #tf.convert_to_tensor(np.eye(10), dtype=tf.float32)
priormu = tf.zeros(n_theta, dtype=tf.float32)


# load theta and data

theta = []
for i in range(10):
    #data.append(np.load(outdir + 'data/data_%d.npy'%(i)))
    theta.append(np.load(datadir + 'data/theta_full_%d.npy'%(i)))

theta = np.concatenate(theta)[:, :n_theta]

# z, mb, dmb^2, x1, dx1^2, c, dc^2, mb/dmb^2, x1/dx1^2, c/dc^2
data = np.load(datadir + 'data/data_full_obs_full_100k.npy')

# stack up the data and parameters
data = tf.convert_to_tensor(np.arcsinh(data), dtype=tf.float32)
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

        plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
        plt.xlabel('$\hat{\Omega}_m - \Omega_m$')
        plt.legend(frameon=False)
        plt.savefig(checkpoint_folder + 'Om_plot.png')
        plt.close()

        plt.hist(mle[:,1].numpy() - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
        plt.xlabel('$\hat{w} - w$')
        plt.legend(frameon=False)
        plt.savefig(checkpoint_folder + 'w_plot.png')
        plt.close()

        np.save(checkpoint_folder + 'mle', mle.numpy())
        np.save(checkpoint_folder + 'F', F.numpy() )
        np.save(checkpoint_folder + 'theta', theta.numpy())


# initialize the Fishnet model
Model = FishnetTwin(n_parameters=n_theta, 
                n_inputs=10, 
                n_hidden_score=[512, 512, 512], 
                activation_score=[tf.nn.elu, tf.nn.elu,  tf.nn.elu],
                n_hidden_fisher=[512, 512, 512], 
                activation_fisher=[tf.nn.elu, tf.nn.elu,  tf.nn.elu],
                optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                theta_fid=theta_fid,
                priormu=tf.zeros(n_theta, dtype=tf.float32),
                priorCinv=tf.eye(n_theta, dtype=tf.float32))

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

# make plots

# model MLEs
mle, F = Model.compute_mle_(data, score_mask, fisher_mask)

# save mles
np.save(outdir + "pred_mle", mle)
np.save(outdir + "pred_F", F)
np.save(outdir + "theta", theta.numpy())

plt.hist(mle[:,0].numpy() - theta[:,0].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.xlabel('$\hat{\Omega}_m - \Omega_m$')
plt.legend(frameon=False)
plt.savefig(outdir + 'Om_plot.png')
plt.close()

plt.hist(mle[:,1].numpy() - theta[:,1].numpy(), bins = 60, histtype='step', density=True, label='learned score MLE')
plt.xlabel('$\hat{w} - w$')
plt.legend(frameon=False)
plt.savefig(outdir + 'w_plot.png')
plt.close()