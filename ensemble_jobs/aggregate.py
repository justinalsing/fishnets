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

# data sizes
n_sims = 10000
n_data = 10000


# fiducial parameters
theta_fid = np.array([0.,0.])

# prior mean and covariance
priorCinv = np.eye(2)
priormu = np.array([0.,0.])

###### load all the data
print("loading data")

datadir = "/data80/makinen/fishnets/" #configs["datapath"]

# small data
datafolder = "ensemble/"

small_data  = np.load(datadir + datafolder + "data.npy")
small_theta = np.load(datadir + datafolder + "theta.npy")

small_F_ = np.load(datadir + datafolder + "F_.npy")
small_t_ = np.load(datadir + datafolder + "t_.npy")
small_pmle_ = np.load(datadir + datafolder + "pmle_.npy")

# BIG data
datafolder = "ensemble_test/"

data = np.load(datadir + datafolder + "data.npy")
theta = np.load(datadir + datafolder + "theta.npy")

print('big data shape', data.shape)


F_ = np.load(datadir + datafolder + "F_.npy")
t_ = np.load(datadir + datafolder + "t_.npy")
pmle_ = np.load(datadir + datafolder + "pmle_.npy")


# where to save the models ?
outdir = "/data80/makinen/fishnets/results/ensemble_training/"

outdirs = [outdir + "model_%d/"%(i) for i in range(10)]


# load dem datas
ensemble_Fs = np.array([np.load(outdirs[i] + "pred_F_test.npy") for i in range(10)])
ensemble_mles = np.array([np.load(outdirs[i] + "pred_mle_test.npy") for i in range(10)])

ensemble_Fs_small = np.array([np.load(outdirs[i] + "pred_F.npy") for i in range(10)])
ensemble_mles_small = np.array([np.load(outdirs[i] + "pred_mle.npy") for i in range(10)])

# compare mse loss for big data and small data

# big data
mse_loss = np.array([np.mean((ensemble_mles[i] - pmle_)**2) for i in range(10)])

# small data
mse_loss_small = np.array([np.mean((ensemble_mles_small[i] - small_pmle_)**2) for i in range(10)])

# weighted average mle for test data
mle = np.average(ensemble_mles, axis=0, weights=1. / mse_loss)

mle_small = np.average(ensemble_mles_small, axis=0, weights= 1. / mse_loss_small)

# save everything 

# for big data
np.save('outputs/big_ensemble_Fs', ensemble_Fs)
np.save('outputs/big_ensemble_mles', ensemble_mles)
np.save('outputs/big_pmle', pmle_)
np.save('outputs/big_average_mle', mle)
np.save('outputs/big_F', F_)
np.save('outputs/big_t', t_)
np.save('outputs/big_data', data)
np.save('outputs/big_theta', theta)

# for small data
np.save('outputs/small_ensemble_Fs', ensemble_Fs_small)
np.save('outputs/small_ensemble_mles', ensemble_mles_small)
np.save('outputs/small_pmle', small_pmle_)
np.save('outputs/small_average_mle', mle_small)
np.save('outputs/small_F', small_F_)
np.save('outputs/small_t', small_t_)
np.save('outputs/small_data', small_data)
np.save('outputs/small_theta', small_theta)


# do pmle plots
plt.hist(mle[:,0] - theta[:,0], bins = 60, histtype='step', density=True, label='ensemble score MLE')
plt.hist(pmle_[:,0] - theta[:,0], bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,0] - theta[:,0])
x = np.linspace(-4*std, 4*std, 500)

plt.xlabel('$\hat{m} - m$')
plt.legend(frameon=False)
plt.show()
plt.savefig('/home/makinen/repositories/fishnets/ensemble_jobs/plots/m_plot_test.png')
plt.close()

plt.hist(mle[:,1] - theta[:,1], bins = 60, histtype='step', density=True, label='ensemble score MLE')
plt.hist(pmle_[:,1] - theta[:,1], bins = 60, histtype='step', density=True, label='exact MLE')
std = np.std(pmle_[:,1] - theta[:,1])
x = np.linspace(-4*std, 4*std, 500)

plt.xlim(-6*std, 6*std)

plt.xlabel('$\hat{c} - c$')
plt.legend(frameon=False)
plt.show()
plt.savefig('/home/makinen/repositories/fishnets/ensemble_jobs/plots/c_plot_test.png')
plt.close()


# do linreg plot for small mse vs big mses
import scipy.stats
import seaborn as sns
sns.set()

linreg = scipy.stats.linregress(x=mse_loss_small, y=mse_loss)

_x = np.linspace(0.0018, 0.004, 50)
_y = _x*linreg.slope + linreg.intercept

sns.regplot(x=mse_loss_small, y=mse_loss)
plt.text(0.0028, 0.0004, s='$y=ax+b$\n$a=$%.2f\n$b=$%.2f'%(linreg.slope, -linreg.intercept))

plt.xlabel('small data mse')
plt.ylabel('big data mse')
plt.savefig('/home/makinen/repositories/fishnets/ensemble_jobs/plots/mse_pred_linear.png')
plt.show()