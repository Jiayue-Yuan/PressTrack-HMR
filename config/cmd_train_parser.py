
import argparse

def parser_train_config():

    parser = argparse.ArgumentParser()

    # dataset & experiment
    parser.add_argument('--dataset_path',
                        default='MIP',
                        type=str,
                        help='dataset path')
    parser.add_argument('--curr_fold',
                        default=1,
                        type=int,
                        help='curr fold for validation')
    parser.add_argument('--exp_mode',
                        default='unseen_group',
                        type=str,
                        help='unseen_subject or unseen_group')
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='curr gpu')
    parser.add_argument('--encoder',
                        default='resnet18',
                        type=str,
                        help='which encoder')
    parser.add_argument('--temp_encoder',
                        default='gru',
                        type=str,
                        help='which sequence model')

    # weights
    parser.add_argument('--e_pressure_weight', default=0., type=float, help='pressure loss weight')
    parser.add_argument('--e_3d_loss_weight', default=300., type=float, help='3d joint loss weight')
    parser.add_argument('--e_pose_loss_weight', default=60., type=float, help='pose loss weight')
    parser.add_argument('--e_shape_loss_weight', default=1, type=float, help='shape loss weight')
    parser.add_argument('--e_trans_loss_weight', default=60., type=float, help='trans loss weight')

    # checkpoints and logging
    parser.add_argument('--logging_path',
                        default='log',
                        type=str,
                        help='dataset path')
    parser.add_argument('--checkpoints_path',
                        default='checkpoints',
                        type=str,
                        help='dataset path')
    parser.add_argument('--test_save_path',
                        default='results/trans_nums',
                        type=str,
                        help='test save path')
    parser.add_argument('--test_best_checkpoints',
                        type=str,
                        help='best checkpoints')


    #opt
    parser.add_argument('--lr',
                        default=1e-6,
                        type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        default=0.02,
                        type=float,
                        help='weight decay')
    parser.add_argument('--cosine',
                        default=True,
                        type=bool,
                        help='cosine lr')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='epochs')
    parser.add_argument('--warmup_epochs',
                        default=5,
                        type=int,
                        help='warm_up epochs')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--val_batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--num_smplify_iters',
                        default=[200, 290],
                        type=list,
                        help='number of iters')
    parser.add_argument('--seqlen',
                        default=16,
                        type=int,
                        help='sequence length')
    parser.add_argument('--overlap',
                        default=0.9,
                        type=float,
                        help='overlap ratio')
    parser.add_argument('--opt',
                        default='AdamW',
                        type=str,
                        help='optimizer')

    parser.add_argument('--note', default='', type=str)

    # network
    parser.add_argument('--img_size',
                        default=(238, 120),
                        help='image size')


    #gru
    parser.add_argument('--gru_hidden_layers', default=512, type=int, help='gru hidden layer length')
    parser.add_argument('--gru_layers', default=1, type=int, help='no. of gru layers')
    parser.add_argument('--trans_depth', default=2, type=int, help='no. of trans layers')

    # regressor
    parser.add_argument('--feature_len', default=768, type=int, help='base encoder feature length')
    parser.add_argument('--tem_feature_len', default=1024, type=int, help='temporal encoder feature length')


    # smpl
    parser.add_argument('--smpl_model',
                        default='smpl',
                        type=str,
                        help='smpl/smplx/smplh')
    parser.add_argument('--smpl_gender',
                        default='neutral',
                        type=str,
                        help='male/female/neutral')

    args = parser.parse_args()

    return args