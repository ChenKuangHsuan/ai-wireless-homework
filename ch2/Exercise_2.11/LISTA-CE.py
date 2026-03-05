"""
Exercise 2.11: LISTA-CE for Channel Estimation

This script implements the Learned Iterative Shrinkage-Thresholding Algorithm 
for Channel Estimation (LISTA-CE) using TensorFlow 1.x. 
It simulates a wideband OFDM-MIMO system and reconstructs the sparse channel 
from compressive measurements.

TODO Implementations Included:
1. Core ISTA update step with residual connections.
2. Layer-wise Mean Squared Error (MSE) loss function.
3. Normalized Mean Square Error (NMSE) calculation.
4. Adam Optimizer definition.
"""

import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as pyplot
from numpy import linalg as la


# ────────────────────────  Experiment Configuration  ───────────────────────── #
parser = argparse.ArgumentParser(description='Parameters for LISTA_CE')
parser.add_argument('--CS_ratio', type=float, default=2, help='Compression ratio')
parser.add_argument('--SNR',      type=float, default=10, help='Signal-to-Noise Ratio (dB)')
parser.add_argument('--layers',   type=int,   default=20, help='Number of unfold layers (iterations)')
parser.add_argument('--Path_num', type=int,   default=3,  help='Number of sparse paths in channel')
args = parser.parse_args()

# Global TensorFlow Settings
reuse   = tf.AUTO_REUSE
fdtype  = tf.float32
tf.reset_default_graph()

# System Parameters (Wideband OFDM-MIMO)
CS_ratio      = args.CS_ratio
CS_ratio_tag  = str(int(CS_ratio * 100)).zfill(3)
complex_label = 2   # 2 for complex signal, 1 for real signal
Nc            = 32  # Number of subcarriers
Nt            = 32  # Number of transmitted antennas
M             = int(complex_label * Nc * Nt * CS_ratio) # Measurement dimension

SNR           = args.SNR
SNR_tag       = str(int(SNR * 10)).zfill(3)
sigma_w       = np.sqrt(1 / np.power(10., SNR / 10.))

Path_num      = args.Path_num

# LISTA Parameters
max_iteration = args.layers
Test_layer    = max_iteration

# Training Parameters
lr               = 0.0001
train_batchsize  = 64
test_batchsize   = 1024
max_episode      = 5000
Test_epoch       = max_episode

# Execution Flags
Train_flag          = 1  # 1 for Training, 0 for Testing
Draw_flag           = 0  # 1 for Draw Picture, 0 for No Draw Picture
Layer_by_layer_flag = 0  # 1 for layer_by_layer, 0 for otherwise
timeline_flag       = 0  # 1 for Draw timeline

# Logging & Saving
type_tag         = f"CS{CS_ratio_tag}_layer{max_iteration}_SNR{SNR_tag}_Path{Path_num}"
appendix_tag     = "_SingleNet"
model_dir        = f"LISTA_CE_{type_tag}{appendix_tag}"
output_file_name = f"Log_output_{model_dir}.txt"


# ────────────────────────  Logging & Utilities  ───────────────────────────── #
def Log_out(Log_out_string, Log_dir=output_file_name):
    """Print to console and append to the log file."""
    print(Log_out_string)
    with open(output_file_name, 'a') as output_file:
        output_file.write(Log_out_string + '\n')

def write_parameters_log(output_file_name, train_flag):
    """Log all hyper-parameters at the beginning of execution."""
    Log_out(f"Running begin at {datetime.now()}")
    Log_out(f"Train_flag = {Train_flag}")
    if train_flag == 1:
        Log_out("### Parameter for wideband OFDM-MIMO system ###")
        Log_out(f"CS_ratio = {CS_ratio}\nSNR = {SNR}\nPath_num = {Path_num}")
        Log_out(f"Num of subcarriers, Nc = {Nc}\nNum of antennas, Nt = {Nt}")
        Log_out("\n### Parameter for LISTA_CE ###")
        Log_out(f"Layer of LISTA_CE = {max_iteration}")
        Log_out("\n### Parameter for Training ###")
        Log_out(f"Learning rate = {lr}\nmodel_dir = {model_dir}")

def phi_gen(init_type="Adaptive selection"):
    """
    Generate the Measurement Matrix (Phi).
    Args:
        init_type (str): Type of generation ('Adaptive selection', 'Gaussion block', 'Gaussion')
    Returns:
        np.ndarray: Measurement matrix of shape [N, M].
    """
    if init_type == "Adaptive selection":
        block_h, block_w = Nt, int(Nt * CS_ratio)
        block = 1. / np.sqrt(block_w) * (2 * np.random.randint(0, 2, size=(block_h, block_w)) - 1)
        Phi_input = np.zeros([block_h * Nc * complex_label, block_w * Nc * complex_label], dtype='float32')
        for index in range(Nc * complex_label):
            Phi_input[index * block_h: (index + 1) * block_h, index * block_w: (index + 1) * block_w] = block
    elif init_type == "Gaussion block":
        block_h, block_w = Nt, int(Nt * CS_ratio)
        block = 1. / np.sqrt(block_w) * np.random.randn(block_h, block_w)
        Phi_input = np.zeros([block_h * Nc * complex_label, block_w * Nc * complex_label], dtype='float32')
        for index in range(Nc * complex_label):
            Phi_input[index * block_h: (index + 1) * block_h, index * block_w: (index + 1) * block_w] = block
    elif init_type == "Gaussion":
        m, n = int(complex_label * Nc * Nt * CS_ratio), int(complex_label * Nc * Nt)
        Phi_input = np.random.randn(n, m) / np.sqrt(m)
    return Phi_input

def load_dataset(Path_num_in):
    """Load Train, Validation, and Test datasets from .mat files."""
    dataset_dir = f'_data_LISTA_CE_BeamFreq_Path{Path_num_in}.mat'
    Log_out(f"Load dataset from Train{dataset_dir}")
    
    Train_data = np.array(scio.loadmat('Train' + dataset_dir)['train_data'])
    Vali_data  = np.array(scio.loadmat('Vali'  + dataset_dir)['vali_data'])
    Test_data  = np.array(scio.loadmat('Test'  + dataset_dir)['test_data'])
    
    Log_out(f"Train_data.shape = {Train_data.shape}")
    Log_out(f"Vali_data.shape = {Vali_data.shape}")
    Log_out(f"Test_data.shape = {Test_data.shape}")
    return Train_data, Vali_data, Test_data


# ────────────────────────  Data & Setup Initialization  ───────────────────── #
Phi_input = phi_gen()
write_parameters_log(output_file_name, Train_flag)
Training_inputs, Vali_inputs, Test_inputs = load_dataset(Path_num)


# ────────────────────────  Model Components (LISTA)  ──────────────────────── #
def variable_w(shape):
    """Create a weight variable with truncated normal initialization."""
    return tf.get_variable('w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1, seed=3))

def variable_b(shape, initial=0.01):
    """Create a bias variable with constant initialization."""
    return tf.get_variable('b', shape=shape, initializer=tf.constant_initializer(initial))

def function_g(x, Phi_tf):
    """Multiply channel matrix with measurement matrix."""
    return tf.matmul(x, Phi_tf)

def function_fs(x, layer_num):
    """Sparsifying Transformation (F) in equ.(12)."""
    input_dim, hidden_dim, output_dim = complex_label * Nt, 128, 256
    
    if layer_num % 2 == 0:  # Trans in equ.(16)
        x = tf.reshape(x, [-1, Nc, input_dim])
        x_left, x_right = x[:, :, 0:Nt], x[:, :, Nt:]
        x_left  = tf.transpose(x_left, perm=[0, 2, 1])
        x_right = tf.transpose(x_right, perm=[0, 2, 1])
        x = tf.concat([x_left, x_right], 2)
    x = tf.reshape(x, [-1, input_dim])
    
    with tf.variable_scope(f'layer_{layer_num}/fs.1', reuse=reuse):
        w, b = variable_w([input_dim, hidden_dim]), variable_b([hidden_dim])
        l = tf.nn.relu(tf.matmul(x, w) + b)
    with tf.variable_scope(f'layer_{layer_num}/fs.2', reuse=reuse):
        w, b = variable_w([hidden_dim, output_dim]), variable_b([output_dim])
        l_out = tf.matmul(l, w) + b
    return l_out

def function_fd(x, layer_num):
    """Invert Transformation (\tilde{F}) in equ.(12)."""
    input_dim, hidden_dim, output_dim = 256, 128, complex_label * Nt
    
    with tf.variable_scope(f'layer_{layer_num}/fd.1', reuse=reuse):
        w, b = variable_w([input_dim, hidden_dim]), variable_b([hidden_dim])
        l = tf.nn.relu(tf.matmul(x, w) + b)
    with tf.variable_scope(f'layer_{layer_num}/fd.2', reuse=reuse):
        w, b = variable_w([hidden_dim, output_dim]), variable_b([output_dim])
        l = tf.matmul(l, w) + b
        
    if layer_num % 2 == 0:  # Trans' in equ.(16)
        l = tf.reshape(l, [-1, Nc, output_dim])
        x_left, x_right = l[:, :, 0:Nt], l[:, :, Nt:]
        x_left  = tf.transpose(x_left, perm=[0, 2, 1])
        x_right = tf.transpose(x_right, perm=[0, 2, 1])
        l = tf.concat([x_left, x_right], 2)
        
    x_recon = tf.reshape(l, [-1, Nc * output_dim])
    return x_recon

def function_soft(x, threshold):
    """Soft denoiser / thresholding operator."""
    return tf.sign(x) * tf.maximum(0., tf.abs(x) - threshold)

def ista_block(Hv, layer_num, W, s, Phi_tf):
    """Construct a single layer of LISTA_CE."""
    with tf.variable_scope(f'layer_{layer_num}', reuse=reuse):
        rho   = tf.Variable(0.15, dtype=fdtype, name='rho')
        theta = tf.Variable(0.15, dtype=fdtype, name='theta')
        
    # ─── TO-DO 1: Implement the core ISTA update step in ista_block ───────── #
    # Remove the NotImplementedError once completed.
    
    raise NotImplementedError("TO-DO 1 is not implemented.")
    
    # ──────────────────────────────────────────────────────────────────────── #
    return Hv_k_output


def inference_ista():
    """Unroll the LISTA network across max_iteration layers."""
    Hv_hat = []
    Hv0 = tf.zeros(tf.shape(Hv_init), dtype=fdtype)
    Hv_hat.append(Hv0)
    
    W = tf.transpose(Phi_tf)
    s = function_g(Hv_init, Phi_tf)
    s = s + tf.random_normal(shape=tf.shape(s), dtype=fdtype) * sigma_w
    
    for i in range(max_iteration):
        Hv_hat_k = ista_block(Hv_hat[-1], i, W, s, Phi_tf)
        Hv_hat.append(Hv_hat_k)
    return Hv_hat

def compute_cost(Hv_hat, Hv_init):
    """Compute MSE loss across all unfolded layers."""
    cost = []
    cost_rec = 0
    for n_layer in range(max_iteration):
        # ─── TO-DO 2: Define the loss function in compute_cost ───────────── #
        # Calculate MSE for the current layer and add it to the cumulative cost (`cost_rec`).
        # Use tf.reduce_mean and tf.square.
        
        raise NotImplementedError("TO-DO 2 is not implemented.")
        
        # ─────────────────────────────────────────────────────────────────── #
        cost.append(cost_rec)
    return cost


# ────────────────────────  Evaluation  ────────────────────────────────────── #
def run_vali(sess, run_type="Vali"):
    """Evaluate NMSE on Validation or Test sets."""
    if run_type == "Vali":
        rand_inds = np.random.choice(Vali_inputs.shape[0], test_batchsize, replace=False)
        batch_xs  = Vali_inputs[rand_inds][:]
    elif run_type == "Test":
        rand_inds = np.random.choice(Test_inputs.shape[0], test_batchsize, replace=False)
        batch_xs  = Test_inputs[rand_inds][:]
    else:
        print("Type error!!!")
        os._exit(1)

    if Train_flag == 0 and Draw_flag == 1:
        print("Draw validation inputs picture")
        input_reshape = np.reshape(batch_xs[0, :], [Nc, 2 * Nt])
        pyplot.imshow(input_reshape)
        pyplot.show()

    # Forward pass to get outputs of all layers
    Hv_out = sess.run([Hv_hat], feed_dict={Hv_init: batch_xs, Phi_tf: Phi_input})
    Hv_out = np.asarray(Hv_out)
    loss_by_layers_NMSE = np.zeros((1, Hv_out.shape[1]))
    
    for ii in range(Hv_out.shape[1]):
        Hv_test_all = Hv_out[0, ii, :, :]
        for jj in range(test_batchsize):
            # ─── TO-DO 3: Calculate the NMSE in run_vali ─────────────────── #
            # Calculate NMSE for a single sample `jj` and accumulate it into `loss_by_layers_NMSE[0, ii]`.
            # Hint: Use np.linalg.norm and np.square.
            
            raise NotImplementedError("TO-DO 3 is not implemented.")
            
            # ─────────────────────────────────────────────────────────────── #
        loss_by_layers_NMSE[0, ii] /= test_batchsize

    if Train_flag == 0:
        print("Draw curve of NMSE by layers")
        x1 = np.linspace(1, loss_by_layers_NMSE.shape[1] - 1, loss_by_layers_NMSE.shape[1] - 1)
        pyplot.semilogy(x1, loss_by_layers_NMSE[0, 1:])
        pyplot.xlabel('Layers')
        pyplot.ylabel('NMSE')
        pyplot.show()
        print(f"loss_by_layers_NMSE = {loss_by_layers_NMSE}")
        
    if Train_flag == 1:
        Log_out(f"loss_by_layers_NMSE = {loss_by_layers_NMSE}")
        
    return loss_by_layers_NMSE


# ────────────────────────  Main Execution (Training)  ─────────────────────── #
if __name__ == "__main__":
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # Placeholders
        Hv_init = tf.placeholder(dtype=fdtype, shape=[None, complex_label * Nc * Nt])
        Phi_tf  = tf.placeholder(dtype=fdtype, shape=[complex_label * Nc * Nt, complex_label * Nc * Nt * CS_ratio])

        # Model Initialization
        Hv_hat   = inference_ista()
        cost_all = compute_cost(Hv_hat, Hv_init)

        with tf.variable_scope('opt', reuse=reuse):
            n_layer = max_iteration - 1
            # ─── TO-DO 4: Define the training operation in the main block ─── #
            # Define the optimizer (`opt`) using tf.train.AdamOptimizer.
            # Minimize `cost_all[n_layer]` and explicitly specify `var_list`.
            
            raise NotImplementedError("TO-DO 4 is not implemented.")
            
            # ──────────────────────────────────────────────────────────────── #

        model  = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
        
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            
            # --- Testing Phase ---
            if Train_flag == 0:
                save_dir = f"{model_dir}/Saved_Model_LISTA_CE_epoch{Test_epoch}"
                model.restore(sess, save_dir)
                run_vali(sess, "Test")              
            
            # --- Training Phase ---
            else:
                loss_episode = []
                run_vali(sess)
                Train_startTime = datetime.now()
                print("Begin training……")
                
                for epoch_i in range(max_episode + 1):
                    epoch_startTime = datetime.now()
                    loss_batch = []
                    
                    # Shuffle training indices
                    rand_inds = np.random.choice(Training_inputs.shape[0], Training_inputs.shape[0], replace=False)
                    num_batches = Training_inputs.shape[0] // train_batchsize
                    
                    for i in range(num_batches):
                        Phi_input = phi_gen()
                        batch_xs  = Training_inputs[rand_inds[i * train_batchsize:(i + 1) * train_batchsize]][:]
                        
                        _, cost_ = sess.run([opt, cost_all[max_iteration - 1]], 
                                            feed_dict={Hv_init: batch_xs, Phi_tf: Phi_input})
                        loss_batch.append(cost_)
                        
                    loss_episode.append(np.mean(loss_batch))
                    
                    # Time tracking & Estimation
                    nowTime        = datetime.now()
                    Train_diffTime = nowTime - Train_startTime
                    epoch_diffTime = nowTime - epoch_startTime
                    restTime       = Train_diffTime / epoch_i * (max_episode - epoch_i) if epoch_i != 0 else Train_diffTime
                    endTime        = nowTime + restTime
                    epoch_time     = f"{epoch_diffTime.seconds}.{epoch_diffTime.microseconds}"

                    # Saving & Logging
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                        
                    if epoch_i <= 1 or epoch_i % 500 == 0:
                        save_dir = f"{model_dir}/Saved_Model_LISTA_CE_epoch{epoch_i}"
                        model.save(sess, save_dir, write_meta_graph=False)
                        
                    if epoch_i % 10 == 0:
                        output_data = (f"layer:[{max_iteration}/{max_iteration}] "
                                       f"epoch:[{epoch_i}/{max_episode}] "
                                       f"cost: {loss_batch[-1]:.5f}, "
                                       f"cost_time: {float(epoch_time):.2f}, "
                                       f"may end at: {endTime}")
                        Log_out(output_data)
                        
                    if epoch_i % 200 == 0 and epoch_i != 0:
                        run_vali(sess)

                # Final Evaluation
                run_vali(sess, "Test")