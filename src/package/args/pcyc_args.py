import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='pcyc_test', help='The directory to save the model and logs')
    parser.add_argument('--sketch_dir', type=str,
                        help='The directory of sketches. The directory can be defined in dataset/data_san.py, but parsed '
                             'value will have the greater priority if given. Whatever, test/training classes of the '
                             'given dataset must be provided in dataset/utils.py.',
                        default='')
    parser.add_argument('--image_dir', type=str,
                        help='The directory of images. The directory can be defined in dataset/data_san.py, but parsed '
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
    parser.add_argument('--paired', type=int, default=0,
                        help='Whether paired data must be given.')
    parser.add_argument('--ni_path', type=str, default=None,
                        help='A pkl file path. The object should be a dict:'
                             ' names[\'class_name\'][\'im\'/\'st\'/\'sk\'][i] = filename_without_postfix_of_the_class_image'
                             'Set it to \'to use default path defined in dataset/data_san.py.')
    parser.add_argument('--lambda-se', default=10.0, type=float, help='Weight on the semantic model')
    parser.add_argument('--lambda-im', default=10.0, type=float, help='Weight on the image model')
    parser.add_argument('--lambda-sk', default=10.0, type=float, help='Weight on the sketch model')
    parser.add_argument('--lambda-gen-cyc', default=1.0, type=float, help='Weight on cycle consistency loss (gen)')
    parser.add_argument('--lambda-gen-adv', default=1.0, type=float, help='Weight on adversarial loss (gen)')
    parser.add_argument('--lambda-gen-cls', default=1.0, type=float, help='Weight on classification loss (gen)')
    parser.add_argument('--lambda-gen-reg', default=0.1, type=float, help='Weight on regression loss (gen)')
    parser.add_argument('--lambda-disc-se', default=0.25, type=float, help='Weight on semantic loss (disc)')
    parser.add_argument('--lambda-disc-sk', default=0.5, type=float, help='Weight on sketch loss (disc)')
    parser.add_argument('--lambda-disc-im', default=0.5, type=float, help='Weight on image loss (disc)')
    parser.add_argument('--lambda-regular', default=0.001, type=float, help='Weight on regularizer')
    parser.add_argument('--dim_enc', default=128, type=int, help='Output dimension of sketch and image')
    parser.add_argument('--print_every', type=int, default=1, help='Number of epochs to print information')
    parser.add_argument('--save_every', type=int, default=1, help='Number of epochs to save model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', default=0.002, metavar='LR',
                        help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--dataset', type=str, default='sketchy', help='[sketchy] or [tuberlin]')
    return parser.parse_args()
