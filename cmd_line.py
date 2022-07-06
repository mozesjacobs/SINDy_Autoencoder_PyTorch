import argparse

args = None

def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # anything that affects the name of the saved folders (for checkpoints, experiments, tensorboard)
    parser.add_argument('-sess', '--session_name', default="june5_speed", type=str, help="session name (KG, HyperKG, HyperKG_RNN)")
    parser.add_argument('-M',  '--model', default="VAE", type=str, help="model to use")
    #parser.add_argument('-sess', '--session_name', default="may27_beta1", type=str, help="session name (KG, HyperKG, HyperKG_RNN)")
    #parser.add_argument('-M',  '--model', default="KG0", type=str, help="model to use")
    #parser.add_argument('-sess', '--session_name', default="may29_beta1", type=str, help="session name (KG, HyperKG, HyperKG_RNN)")
    #parser.add_argument('-M',  '--model', default="VAE", type=str, help="model to use")
    #parser.add_argument('-sess', '--session_name', default="may30_beta1", type=str, help="session name (KG, HyperKG, HyperKG_RNN)")
    #parser.add_argument('-M',  '--model', default="KG0_2", type=str, help="model to use")
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")
    parser.add_argument('-LA', '--labels', default=None, type=list, help="Which labels to train on (None for all)")
    parser.add_argument('-DE', '--decoder', default="12", type=str, help="Which decoder to use. See models/decoder.py")
    parser.add_argument('-DT', '--data_set', default='smooth_moving_mnist', type=str, help="Which dataset to use (smooth_moving_mnist, ucf101, sprites)")
    parser.add_argument('-DV', '--data_vers', default="set3", type=str, help="Which dataset version to use for MNIST (str None if sprites)")

    # model parameters
    parser.add_argument('-NG', '--decoder_channels', default=16, type=int, help="Factor for dim of channels inside decoder")
    parser.add_argument('-BN', '--batch_norm', default=False, type=int, help="If true, uses batch norm in decoder")
    parser.add_argument('-R', '--r_dim', default=256, type=int, help="Dimension of r")
    parser.add_argument('-NL', '--num_mlp_layers', default=2, type=int, help="Number of layers in MLP")
    parser.add_argument('-R2', '--r2_dim', default=8, type=int, help="Dimension of r2")
    parser.add_argument('-R3', '--r3_dim', default=10, type=int, help="Dimension of r3")
    parser.add_argument('-RW', '--r_w', default=8, type=int, help="Dimension of r width")
    parser.add_argument('-RH', '--r_h', default=8, type=int, help="Dimension of r height")
    parser.add_argument('-RL', '--recon_loss', default="BCE", type=str, help="Which reconstruction loss function to use (BCE, MSE)")
    parser.add_argument('-H', '--hidden_dim', default=320, type=int, help="Dimension of LSTM outpust")
    parser.add_argument('-LH', '--lstm_hidden_dim', default=320, type=int, help="Dimension of hypernet lstm")
    parser.add_argument('-CGG', '--create_grad_graph', default=1, type=int, help="Whether create_graph = True")
    parser.add_argument('-LC', '--latent_channels', default=320, type=int, help="Number of channels in latent r")

    # training / testing
    parser.add_argument('-D', '--device', default=1, type=int, help="Which device to use")
    parser.add_argument('-E', '--epochs', default=50, type=int, help="Number of Training Epochs")
    parser.add_argument('-B', '--batch_size', default=50, type=int, help="Batch size")    
    parser.add_argument('-T', '--timesteps', default=20, type=int, help="How many timesteps to run on")
    parser.add_argument('-I', '--checkpoint_interval', default=1, type=int, help="Saves the model every checkpoint_interval intervals")
    parser.add_argument('-TI', '--test_interval', default=101, type=int, help="Tests the model every test_interval intervals")
    parser.add_argument('-IGI', '--inference_grid_interval', default=100, type=int, help="Runs inference grid every -IGI epochs")
    parser.add_argument('-SGI', '--sample_grid_interval', default=100, type=int, help="Runs sample grid every -SGI epochs")
    parser.add_argument('-SGCI', '--sample_context_grid_interval', default=100, type=int, help="Runs sample grid every -SGI epochs")
    parser.add_argument('-CL', '--clip', default=1.0, type=float, help="Gradient clip value")
    parser.add_argument('-GF', '--gamma_factor', default=0.9995, type=float, help="Learning rate decay factor")
    parser.add_argument('-C', '--load_cp', default=0, type=int, help="If 1, loads previous checkpoint")
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-lmda', '--regularization', default=1e-4, type=float, help="L2 penalty")
    parser.add_argument('-IE', '--initial_e', default=0, type=int, help="Initial Number of Epochs")
    parser.add_argument('-BM', '--beta_max', default=1.0, type=float, help="Max beta value for KL")
    parser.add_argument('-W', '--weight_update', default=0.1, type=float, help="KLD epoch weight update")
    parser.add_argument('-SW', '--specific_weight', default=None, type=float, help="KLD epoch weight (if None, uses defaults with weight_update")
    parser.add_argument('-IKLD', '--init_kld', default=1.0, type=float, help="Initial KLD")

    # data
    parser.add_argument('-F', '--frame_size', default=64, type=int, help="Dimensions of the frames")
    parser.add_argument('-CH', '--channels', default=1, type=int, help="Number of channels in image frame")
    parser.add_argument('-SEQ', '--seq_len', default=20, type=int, help="Number of frames in a sequence")
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # experiment arguments  
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # misc
    parser.add_argument("--gif_frame_duration", default=0.25, type=float, help="Duration of each frame in the gif")
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help="If 1, prints the name of the experiments output folder")
    
    # grids / graphs
    parser.add_argument("--sample_grid", default=0, type=int, help="Sample grid test")
    parser.add_argument("--sample_context_grid", default=0, type=int, help="Sample with context frames grid test")
    parser.add_argument('--num_context', default=5, type=int, help="Number of context frames to use")
    parser.add_argument("--inference_grid", default=0, type=int, help="Inference grid test")
    parser.add_argument("--num_sample_grid", default=5, type=int, help="Number of samples to plot in sample grid test")
    parser.add_argument("--num_inference_grid", default=5, type=int, help="Number of samples to plot in inference grid test")
    parser.add_argument("--num_sample_context_grid", default=10, type=int, help="Number of samples to plot in sample context grid test with")
    
    return parser.parse_args() 

def run_args():
    global args
    if args is None:
        args = parse_args()


run_args()