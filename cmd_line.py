import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # anything that affects the name of the saved folders (for checkpoints, experiments, tensorboard)
    # path saved as: initial_folder/data_set/model/session_name
    # where initial_folder is experiments, model_folder, or tensorboard_folder
    parser.add_argument('-sess', '--session_name', default="june11", type=str, help="Session name")
    parser.add_argument('-M',  '--model', default="SINDyAE", type=str, help="Model to use")
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")
    parser.add_argument('-DT', '--data_set', default='lorenz', type=str, help="Which dataset to use (lorenz)")
    
    # network parameters
    parser.add_argument('-Z', '--z_dim', default=3, type=int, help="Size of latent vector")
    parser.add_argument('-U',  '--u_dim', default=128, type=int, help="Sise of u vector in Lorenz data")
    parser.add_argument('-HD', '--hidden_dim', default=64, type=str, help="Dimension of hidden layers in autoencoder")
    parser.add_argument('-PO', '--poly_order', default=3, type=str, help="Highest polynomial degree to include in library")
    parser.add_argument('-US', '--use_sine', default=False, type=bool, help="Iff true, includes sine function in library")
    parser.add_argument('-IC', '--include_constant', default=True, type=bool, help="Iff true, includes constant term in library")
    parser.add_argument('-NL', '--nonlinearity', default='elu', type=str, help="Nonlinearity to use in autoencoder (elu, sig, relu, None)")
    parser.add_argument('-MO', '--model_order', default=1, type=int, help="Determines whether model predits 1st or 2nd order derivatives (values: 1, 2)")
    
    # training parameters
    parser.add_argument('-E', '--epochs', default=100, type=float, help="Number of epochs to train for")
    parser.add_argument('-LR', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-ARE', '--adam_regularization', default=1e-5, type=float, help="Regularization to use in ADAM optimizer")
    parser.add_argument('-GF', '--gamma_factor', default=0.995, type=float, help="Learning rate decay factor")
    parser.add_argument('-BS', '--batch_size', default=50, type=float, help="Batch size")
    parser.add_argument('-L1', '--lambda_dx', default=1e-4, type=float, help="Weight of dx loss")
    parser.add_argument('-L2', '--lambda_dz', default=0, type=float, help="Weight of dz loss")
    parser.add_argument('-L3', '--lambda_reg', default=1e-5, type=float, help="Weight of regularization loss")
    parser.add_argument('-C', '--clip', default=None, type=float, help="Gradient clipping value during training (None for no clipping)")
    parser.add_argument('-TI', '--test_interval', default=1, type=float, help="Epoch interval to evaluate on val (test) data during training")
    parser.add_argument('-CPI', '--checkpoint_interval', default=1, type=float, help="Epoch interval to save model during training")
    parser.add_argument('-ST', '--sequential_threshold', default=5e-2, type=float, help="Sequential thresholding value for coefficients")

    # lorenz dataset parameters
    parser.add_argument('-TIC', '--train_initial_conds', default=2048, type=int, help='Number of initial conditions in the training set')
    parser.add_argument('-VIC', '--val_initial_conds', default=20, type=int, help='Number of initial conditions in the validation set')
    parser.add_argument('-TEIC', '--test_initial_conds', default=100, type=int, help='Number of initial conditions in the test set')
    parser.add_argument('-NSTR', '--noise_strength', default=0, type=float, help='Strength of noise in lorenz datasaet. 0 for no noise')
    parser.add_argument('-TS', '--timesteps', default=250, type=int, help='Number of timesteps per trajectory')

    # other
    parser.add_argument('-LCP', '--load_cp', default=0, type=int, help='If 1, loads the model from the checkpoint. If 0, does not')
    parser.add_argument('-D', '--device', default=1, type=int, help='Which GPU to use')
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help='Iff true, prints the folder for different logs')

    return parser.parse_args() 