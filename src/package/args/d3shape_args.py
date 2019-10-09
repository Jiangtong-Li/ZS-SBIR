import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='d3shape_test', help='The directory to save the model and logs')
    parser.add_argument('--cp', type=float, default=0.2,
                        help='Parameter Cn in the Siamese-2 loss')
    parser.add_argument('--cn', type=float, default=10,
                        help='Parameter Cn in the Siamese-2 loss')
    parser.add_argument('--sketch_dir', type=str,
                        help='The directory of sketches. The directory can be defined in dataset/data_d3shape.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--imsk_dir', type=str,
                        help='The directory of sketch tokens of the images. The directory can be defined in dataset/data_d3shape.py, but parsed '
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
    parser.add_argument('--print_every', type=int, default=100, help='Number of steps to print information')
    parser.add_argument('--save_every', type=int, default=1000, help='Number of steps to save model')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--l2_reg', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=1e5, help='Number of steps to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='sketchy', help='[sketchy] or [tuberlin]')
    return parser.parse_args()
