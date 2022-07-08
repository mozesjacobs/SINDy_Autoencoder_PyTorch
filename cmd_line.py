import argparse

args = None

def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # anything that affects the name of the saved folders (for checkpoints, experiments, tensorboard)
    parser.add_argument('-sess', '--session_name', default="june5_speed", type=str, help="session name")
    parser.add_argument('-M',  '--model', default="SINDyAE", type=str, help="model to use")
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")
    parser.add_argument('-DT', '--data_set', default='lorenz', type=str, help="Which dataset to use (lorenz)")
    
    # network parameters
    parser.add_argument('-Z', '--z_dim', default=3, type=int, help="Size of latent vector")
    parser.add_argument('-U',  '--u_dim', default=128, type=int, help="Sise of u vector in Lorenz data")
    parser.add_argument('-HD', '--hidden_dim', default=64, type=str, help="Dimmension of hidden layers in autoencoder")
    parser.add_argument('-PO', '--poly_order', default=3, type=str, help="Highest polynomial degree to include in library")
    parser.add_argument('-US', '--use_sine', default=False, type=bool, help="Iff true, includes sine function in library")
    parser.add_argument('-IC', '--include_constant', default=True, type=bool, help="Iff true, includes constant term in library")
    
    # training parameters
    parser.add_argument('-LR', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-BS', '--batch_size', default=50, type=float, help="Batch size")
    parser.add_argument('-L1', '--lambda_1', default=1e-4, type=float, help="Weight of dx loss")
    parser.add_argument('-L2', '--lambda_2', default=1e-3, type=float, help="Weight of dz loss")
    parser.add_argument('-L3', '--lambda_3', default=1e-5, type=float, help="Weight of regularization loss")

    # lorenz dataset parameters
    parser.add_argument('-TIC', '--train_initial_conds', default=2048, type=int, help='Number of initial conditions in the training set')
    parser.add_argument('-VIC', '--val_initial_conds', default=20, type=int, help='Number of initial conditions in the validation set')
    parser.add_argument('-TEIC', '--test_initial_conds', default=100, type=int, help='Number of initial conditions in the test set')
    parser.add_argument('-NSTR', '--noise_strength', default=0, type=float, help='Strength of noise in lorenz datasaet. 0 for no noise')

    return parser.parse_args() 