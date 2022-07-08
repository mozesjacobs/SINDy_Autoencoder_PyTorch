import os
import numpy as np
from cmd_line import parse_args
from src.dataset.lorenz_functions import get_lorenz_data
from src.utils.other import get_lorenz_path


def main():
    # get args
    args = parse_args()
    
    # create data
    train_data = get_lorenz_data(n_ics=args.train_initial_conds, noise_strength=args.noise_strength)
    val_data = get_lorenz_data(n_ics=args.val_initial_conds, noise_strength=args.noise_strength)
    test_data = get_lorenz_data(n_ics=args.test_initial_conds, noise_strength=args.noise_strength)

    # save data
    data_paths = get_lorenz_path()
    np.save(data_paths[0], train_data)
    np.save(data_paths[1], val_data)
    np.save(data_paths[2], test_data)
    

if __name__ == '__main__':
    main()