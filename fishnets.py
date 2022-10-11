import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
tfk = tf.keras

# fishnet class (single network model for both score and fisher)
class Fishnet(tf.Module):
    
    def __init__(self, n_parameters=2, n_inputs=3, n_hidden=[128, 128], activation=[tf.nn.leaky_relu, tf.nn.leaky_relu], priormu=None, priorCinv=None, theta_fid=None, optimizer=tf.keras.optimizers.Adam(lr=1e-4), maxcall=1e6):
        
        # parameters
        self.n_parameters = n_parameters
        self.n_inputs = n_inputs
        self.architecture = [n_inputs] + n_hidden + [n_parameters + n_parameters * (n_parameters + 1) // 2]
        self.n_layers = len(self.architecture) - 1
        self.activation = activation + [tf.identity]
        self.optimizer = optimizer
        self.maxcall = int(maxcall)
        self.priormu = priormu
        self.priorCinv = priorCinv
        self.theta_fid = theta_fid
        
        # model
        self.model = tfk.Sequential([tfk.layers.Dense(self.architecture[i+1], activation=self.activation[i]) for i in range(self.n_layers)])
        _ = self.model(tf.ones((1, 1, n_inputs)))
        
        # set up for l-bfgs optimizer...
        
        # trainable variables shapes
        self.trainable_variable_shapes = tf.shape_n(self.model.trainable_variables)
        
        # prepare stich and partition indices for dynamic stitching and partitioning
        count = 0
        stitch_idx = [] # stitch indices
        partition_idx = [] # partition indices

        for i, shape in enumerate(self.trainable_variable_shapes):
            n = np.product(shape)
            stitch_idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            partition_idx.extend([i]*n)
            count += n

        self.partition_idx = tf.constant(partition_idx)
        self.stitch_idx = stitch_idx
        
    # for l-bfgs optimizer: assign model parameters from a 1d tensor
    @tf.function
    def assign_new_model_parameters(self, parameters_1d):

        parameters = tf.dynamic_partition(parameters_1d, self.partition_idx, len(self.trainable_variable_shapes))
        for i, (shape, param) in enumerate(zip(self.trainable_variable_shapes, parameters)):
            self.model.trainable_variables[i].assign(tf.reshape(param, shape))
            
    # run lbfgs optimizer
    def lbfgs_optimize(self, inputs, parameters, score_mask, fisher_mask, max_iterations=500, tolerance=1e-5, verbose=True):
        
        # initial parameters
        initial_parameters = tf.dynamic_stitch(self.stitch_idx, self.model.trainable_variables)
        
        # value and gradients function, as required by l-bfgs
        def value_and_gradient(x):
            
            # set the updated network parameters
            self.assign_new_model_parameters(x)
            
            # compute loss and gradients (using gradient accumulation if needed)
            loss, gradients = self.compute_loss_and_gradients(inputs, parameters, score_mask, fisher_mask)
            gradients = tf.dynamic_stitch(self.stitch_idx, gradients) # stitch the gradients together into 1d, as needed by l-bfgs
            
            # print the loss if desired
            if verbose:
                print(loss.numpy())
            
            return loss, gradients
        
        # run the optimizer
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=value_and_gradient,
                                              initial_position=initial_parameters,
                                              max_iterations=max_iterations,
                                              tolerance=tolerance)
        
        return results
        
    # compute score and fisher outputs from the network model
    @tf.function
    def call(self, inputs):
        
        score, fisher_cholesky = tf.split(self.model(inputs), (self.n_parameters, self.n_parameters * (self.n_parameters + 1) // 2), axis=-1)
        
        return score, self.construct_fisher_matrix(fisher_cholesky)
    
    # construct the Fisher matrix from the network outputs (ie., the elements of the cholseky factor)
    @tf.function
    def construct_fisher_matrix(self, outputs):
        
        Q = tfp.math.fill_triangular(outputs)
        L = Q - tf.linalg.diag(tf.linalg.diag_part(Q) + tf.math.softplus(tf.linalg.diag_part(Q)))
        return tf.einsum('...ij,...jk->...ik', L, tf.transpose(L, perm=[0, 1, 3, 2]))
    
    # construct the MLE
    @tf.function
    def compute_mle(self, inputs, score_mask, fisher_mask):
        
        # score and fisher (per data point)
        score, fisher = self.call(inputs)
        
        # sum the per-data point scores and Fishers, and include Gaussian prior correction
        t = tf.reduce_sum(score*score_mask, axis=1) - tf.einsum('ij,j->i', self.priorCinv, (self.theta_fid - self.priormu))
        F = tf.reduce_sum(fisher*fisher_mask, axis=1) + self.priorCinv
        
        # construct MLE
        mle = self.theta_fid + tf.einsum('ijk,ik->ij', tf.linalg.inv(F), t)
    
        return mle, F
    
    # automatically break up the calculation if massive batches are called (to avoid memory issues)
    def compute_mle_(self, inputs, score_mask, fisher_mask):
        
        # total number of network calls
        ncalls = inputs.shape[0] * inputs.shape[1]
        
        # do we need to split the call into batches or not?
        if ncalls > self.maxcall:
            
            # batch up the inputs
            batch_size = (self.maxcall // inputs.shape[1])
            idx = [np.arange(i*batch_size, min((i+1)*batch_size, inputs.shape[0])) for i in range(inputs.shape[0] // batch_size + int(inputs.shape[0] % batch_size > 0))]
            
            # compute the MLE and fisher over batches
            mle = np.zeros((inputs.shape[0], self.n_parameters))
            F = np.zeros((inputs.shape[0], self.n_parameters, self.n_parameters))
            for i in trange(len(idx)):
                mle_, F_ = Model.compute_mle(data.numpy()[idx[i],...], score_mask.numpy()[idx[i],...], fisher_mask.numpy()[idx[i],...] )
                mle[idx[i],:] = mle_.numpy()
                F[idx[i],...] = F_.numpy()
            mle = tf.convert_to_tensor(mle)
            F = tf.convert_to_tensor(F)
        else:
        
            # compute MLE and fisher
            mle, F = self.compute_mle(inputs, score_mask, fisher_mask)
    
        return mle, F  

    # KL divergence loss
    @tf.function
    def kl_loss(self, inputs, parameters, score_mask, fisher_mask):
        
        mle, F = self.compute_mle(inputs, score_mask, fisher_mask)
    
        return -tf.reduce_mean(-0.5 * tf.einsum('ij,ij->i', (parameters - mle), tf.einsum('ijk, ik -> ij', F, (parameters - mle))) + 0.5*tf.linalg.logdet(F))

    # basic loss and gradients function
    @tf.function
    def loss_and_gradients(self, inputs, parameters, score_mask, fisher_mask):
        
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.kl_loss(inputs, parameters, score_mask, fisher_mask)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        return loss, gradients
        
    
    def compute_loss_and_gradients(self, inputs, parameters, score_mask, fisher_mask):
        
        # total number of network calls
        ncalls = inputs.shape[0] * inputs.shape[1]
        
        # accumulate gradients or not?
        if ncalls > self.maxcall:
            
            # how many batches do we need to split it into?
            batch_size = (self.maxcall // inputs.shape[1])
            
            # create dataset to do sub-calculations over
            dataset = tf.data.Dataset.from_tensor_slices((inputs, parameters, score_mask, fisher_mask)).batch(batch_size)

            # initialize gradients and loss (to zero)
            accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.model.trainable_variables]
            accumulated_loss = tf.Variable(0., trainable=False)

            # loop over sub-batches
            for inputs_, parameters_, score_mask_, fisher_mask_ in dataset:

                # calculate loss and gradients
                loss, gradients = self.loss_and_gradients(inputs_, parameters_, score_mask_, fisher_mask_)

                # update the accumulated gradients and loss
                for i in range(len(accumulated_gradients)):
                    accumulated_gradients[i].assign_add(gradients[i]*inputs_.shape[0]/inputs.shape[0])
                accumulated_loss.assign_add(loss*inputs_.shape[0]/inputs.shape[0])

        else:
            loss, gradients = self.loss_and_gradients(inputs, parameters, score_mask, fisher_mask)

        return loss, gradients
    
    def train(self, training_data, epochs=1000, lr=None, batch_size=512):
        
        if lr is not None:
            self.optimizer.lr = lr
            
        dataset = tf.data.Dataset.from_tensor_slices(training_data)
        with trange(epochs) as progress:
            for epoch in progress:
                for inputs_, parameters_, score_mask_, fisher_mask_ in dataset.shuffle(buffer_size=len(dataset)).batch(batch_size):
                    loss, gradients = self.compute_loss_and_gradients(inputs_, parameters_, score_mask_, fisher_mask_)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    progress.set_postfix({'loss':loss.numpy()})
                    
                    
# fishnet class (seperate network models for both score and fisher)
class FishnetTwin(tf.Module):
    
    def __init__(self, n_parameters=2, n_inputs=3, n_hidden_score=[128, 128], activation_score=[tf.nn.leaky_relu, tf.nn.leaky_relu], n_hidden_fisher=[128, 128], activation_fisher=[tf.nn.leaky_relu, tf.nn.leaky_relu], priormu=None, priorCinv=None, theta_fid=None, optimizer=tf.keras.optimizers.Adam(lr=1e-4), maxcall=1e6):
        
        # parameters
        self.n_parameters = n_parameters
        self.n_inputs = n_inputs
        self.maxcall = int(maxcall)

        # architectures
        self.architecture_score = [n_inputs] + n_hidden_score + [n_parameters]
        self.architecture_fisher = [n_inputs] + n_hidden_fisher + [int(n_parameters * (n_parameters + 1)) // 2]
        self.n_layers_score = len(self.architecture_score) - 1
        self.n_layers_fisher = len(self.architecture_fisher) - 1
        self.activation_score = activation_score + [tf.identity]
        self.activation_fisher = activation_fisher + [tf.identity]

        # optimizer
        self.optimizer = optimizer
        
        # prior
        self.priormu = priormu
        self.priorCinv = priorCinv
        self.theta_fid = theta_fid
        
        # score model
        self.model_score = tfk.Sequential([tfk.layers.Dense(self.architecture_score[i+1], activation=self.activation_score[i]) for i in range(self.n_layers_score)])
        _ = self.model_score(tf.ones((1, 1, n_inputs)))
        
        # fisher model
        self.model_fisher = tfk.Sequential([tfk.layers.Dense(self.architecture_fisher[i+1], activation=self.activation_fisher[i]) for i in range(self.n_layers_fisher)])
        _ = self.model_fisher(tf.ones((1, 1, n_inputs)))
        
        # set up for l-bfgs optimizer...
        
        # trainable variables shapes
        self.trainable_variable_shapes = tf.shape_n(self.trainable_variables)
        
        # prepare stich and partition indices for dynamic stitching and partitioning
        count = 0
        stitch_idx = [] # stitch indices
        partition_idx = [] # partition indices

        for i, shape in enumerate(self.trainable_variable_shapes):
            n = np.product(shape)
            stitch_idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            partition_idx.extend([i]*n)
            count += n

        self.partition_idx = tf.constant(partition_idx)
        self.stitch_idx = stitch_idx
        
    # for l-bfgs optimizer: assign model parameters from a 1d tensor
    @tf.function
    def assign_new_model_parameters(self, parameters_1d):

        parameters = tf.dynamic_partition(parameters_1d, self.partition_idx, len(self.trainable_variable_shapes))
        for i, (shape, param) in enumerate(zip(self.trainable_variable_shapes, parameters)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))
            
    # run lbfgs optimizer
    def lbfgs_optimize(self, inputs, parameters, score_mask, fisher_mask, max_iterations=500, tolerance=1e-5, verbose=True):
        
        # initial parameters
        initial_parameters = tf.dynamic_stitch(self.stitch_idx, self.trainable_variables)
        
        # value and gradients function, as required by l-bfgs
        def value_and_gradient(x):
            
            # set the updated network parameters
            self.assign_new_model_parameters(x)
            
            # compute loss and gradients (using gradient accumulation if needed)
            loss, gradients = self.compute_loss_and_gradients(inputs, parameters, score_mask, fisher_mask)
            gradients = tf.dynamic_stitch(self.stitch_idx, gradients) # stitch the gradients together into 1d, as needed by l-bfgs
            
            # print the loss if desired
            if verbose:
                print(loss.numpy())
            
            return loss, gradients
        
        # run the optimizer
        results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=value_and_gradient,
                                              initial_position=initial_parameters,
                                              max_iterations=max_iterations,
                                              tolerance=tolerance)
        
        return results
        
    # compute score and fisher outputs from the network model
    @tf.function
    def call(self, inputs):
        
        score = self.model_score(inputs)
        fisher_cholesky = self.model_fisher(inputs)
        
        return score, self.construct_fisher_matrix(fisher_cholesky)
    
    # construct the Fisher matrix from the network outputs (ie., the elements of the cholseky factor)
    @tf.function
    def construct_fisher_matrix(self, outputs):
        
        Q = tfp.math.fill_triangular(outputs)
        L = Q - tf.linalg.diag(tf.linalg.diag_part(Q) + tf.math.softplus(tf.linalg.diag_part(Q)))
        return tf.einsum('...ij,...jk->...ik', L, tf.transpose(L, perm=[0, 1, 3, 2]))
    
    # construct the MLE
    @tf.function
    def compute_mle(self, inputs, score_mask, fisher_mask):
        
        # score and fisher (per data point)
        score, fisher = self.call(inputs)
        
        # sum the per-data point scores and Fishers, and include Gaussian prior correction
        t = tf.reduce_sum(score*score_mask, axis=1) - tf.einsum('ij,j->i', self.priorCinv, (self.theta_fid - self.priormu))
        F = tf.reduce_sum(fisher*fisher_mask, axis=1) + self.priorCinv
        
        # construct MLE
        mle = self.theta_fid + tf.einsum('ijk,ik->ij', tf.linalg.inv(F), t)
    
        return mle, F
    
    # automatically break up the calculation if massive batches are called (to avoid memory issues)
    def compute_mle_(self, inputs, score_mask, fisher_mask):
        
        # total number of network calls
        ncalls = inputs.shape[0] * inputs.shape[1]
        
        # do we need to split the call into batches or not?
        if ncalls > self.maxcall:
            
            # batch up the inputs
            batch_size = (self.maxcall // inputs.shape[1])
            idx = [np.arange(i*batch_size, min((i+1)*batch_size, inputs.shape[0])) for i in range(inputs.shape[0] // batch_size + int(inputs.shape[0] % batch_size > 0))]
            
            # compute the MLE and fisher over batches
            mle = np.zeros((inputs.shape[0], self.n_parameters))
            F = np.zeros((inputs.shape[0], self.n_parameters, self.n_parameters))
            for i in trange(len(idx)):
                mle_, F_ = Model.compute_mle(data.numpy()[idx[i],...], score_mask.numpy()[idx[i],...], fisher_mask.numpy()[idx[i],...] )
                mle[idx[i],:] = mle_.numpy()
                F[idx[i],...] = F_.numpy()
            mle = tf.convert_to_tensor(mle)
            F = tf.convert_to_tensor(F)
        else:
        
            # compute MLE and fisher
            mle, F = self.compute_mle(inputs, score_mask, fisher_mask)
    
        return mle, F  

    # KL divergence loss
    @tf.function
    def kl_loss(self, inputs, parameters, score_mask, fisher_mask):
        
        mle, F = self.compute_mle(inputs, score_mask, fisher_mask)
    
        return -tf.reduce_mean(-0.5 * tf.einsum('ij,ij->i', (parameters - mle), tf.einsum('ijk, ik -> ij', F, (parameters - mle))) + 0.5*tf.linalg.logdet(F))

    # basic loss and gradients function
    @tf.function
    def loss_and_gradients(self, inputs, parameters, score_mask, fisher_mask):
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss = self.kl_loss(inputs, parameters, score_mask, fisher_mask)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        return loss, gradients
        
    
    def compute_loss_and_gradients(self, inputs, parameters, score_mask, fisher_mask):
        
        # total number of network calls
        ncalls = inputs.shape[0] * inputs.shape[1]
        
        # accumulate gradients or not?
        if ncalls > self.maxcall:
            
            # how many batches do we need to split it into?
            batch_size = (self.maxcall // inputs.shape[1])
            
            # create dataset to do sub-calculations over
            dataset = tf.data.Dataset.from_tensor_slices((inputs, parameters, score_mask, fisher_mask)).batch(batch_size)

            # initialize gradients and loss (to zero)
            accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
            accumulated_loss = tf.Variable(0., trainable=False)

            # loop over sub-batches
            for inputs_, parameters_, score_mask_, fisher_mask_ in dataset:

                # calculate loss and gradients
                loss, gradients = self.loss_and_gradients(inputs_, parameters_, score_mask_, fisher_mask_)

                # update the accumulated gradients and loss
                for i in range(len(accumulated_gradients)):
                    accumulated_gradients[i].assign_add(gradients[i]*inputs_.shape[0]/inputs.shape[0])
                accumulated_loss.assign_add(loss*inputs_.shape[0]/inputs.shape[0])

        else:
            loss, gradients = self.loss_and_gradients(inputs, parameters, score_mask, fisher_mask)

        return loss, gradients
    
    def train(self, training_data, epochs=1000, lr=None, batch_size=512):
        
        if lr is not None:
            self.optimizer.lr = lr
            
        dataset = tf.data.Dataset.from_tensor_slices(training_data)
        with trange(epochs) as progress:
            for epoch in progress:
                for inputs_, parameters_, score_mask_, fisher_mask_ in dataset.shuffle(buffer_size=len(dataset)).batch(batch_size):
                    loss, gradients = self.compute_loss_and_gradients(inputs_, parameters_, score_mask_, fisher_mask_)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    progress.set_postfix({'loss':loss.numpy()})