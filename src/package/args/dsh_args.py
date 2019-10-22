import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='dsh_test', help='The directory to save the model and logs')
    parser.add_argument('--lamb', type=float, default=0.01,
                        help='The second term of the loss')
    parser.add_argument('--gamma', type=float, default=0.00001,
                        help='The third term of the loss')
    parser.add_argument('--m', type=int, default=128,
                        help='Length of the hash codes')
    parser.add_argument('--sketch_dir', type=str,
                        help='The directory of sketches. The directory can be defined in dataset/data_dsh.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--path_semantic', type=str,
                        help='The directory of semantics, a dict object. The directory can be defined in dataset/data_dsh.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')# config
    parser.add_argument('--config', type=int,
                        help='Configure of the network: 1/2',
                        default=1)
    parser.add_argument('--im_dir', type=str,
                        help='The directory of images of the images. The directory can be defined in dataset/data_dsh.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--imsk_dir', type=str,
                        help='The directory of sketch tokens of the images. The directory can be defined in dataset/data_dsh.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--npy_dir', type=str,
                        help='The npy files directory. By set it 0 to load the default folder provided in '
                             'dataset/data_san.py. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default=None)
    parser.add_argument('--start_from', type=str, default=None,
                        help='The iteration the training starts from or the exact model file name. '
                             'Default: None -- Try to start from the latest checkpoint.')
    parser.add_argument('--update_every', type=int, default=1,
                        help='Update the parameters every certain steps.')
    parser.add_argument('--print_every', type=int, default=200, help='Number of steps to print information')
    parser.add_argument('--save_every', type=int, default=2000, help='Number of steps to save model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--steps', type=int, default=1e5, help='Number of steps to train')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='sketchy', help='[sketchy] or [tuberlin]')
    return parser.parse_args()
