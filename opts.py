import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", default="futr", help='model type')
parser.add_argument("--mode", default="train_eval", help="select action: [\"train\", \
                    \"predict\", \"train_eval\"]")
parser.add_argument("--dataset", type=str, default='breakfast')
parser.add_argument('--predict', "-p", action='store_true', help="predict for whole videos mode")
parser.add_argument('--wandb', type=str, default='project name', help="wandb runs name")

#Dataset
#breakfast
parser.add_argument("--mapping_file", default="./datasets/breakfast/mapping.txt")
parser.add_argument("--features_path", default="./datasets/breakfast/features/")
parser.add_argument("--gt_path", default="./datasets/breakfast/groundTruth/")
parser.add_argument("--split", default="1", help='split number')
parser.add_argument("--file_path", default="./datasets/breakfast/splits")
parser.add_argument("--model_save_path", default="./save_dir/models/transformer")
parser.add_argument("--results_save_path", default="./save_dir/results/transformer")
parser.add_argument("--task", type=str, help="Next Action Anticipation/long-term anticipation", default='long')

#Training options
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--workers", type=int, default= 30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_mul", type=float, default=2.0)
parser.add_argument("--weight_decay", type=float, default=5e-3) #5e-3
parser.add_argument("-warmup", '--n_warmup_steps', type=int, default=500)
parser.add_argument("--cpu", action='store_true', help='run in cpu')
parser.add_argument("--sample_rate", type=int, default=3)
parser.add_argument("--obs_perc", default=30)
parser.add_argument("--n_query", type=int, default=8)


#FUTR specific parameters
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--n_encoder_layer", type=int, default=2)
parser.add_argument("--n_decoder_layer", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--input_dim", type=int, default=2048)

#Model parameters
parser.add_argument("--seg", action='store_true', help='action segmentation')
parser.add_argument("--anticipate", action='store_true', help='future anticipation')
parser.add_argument("--pos_emb", action='store_true', help='positional embedding')
parser.add_argument("--max_pos_len", type=int, default=2000, help='position embedding number for linear interpolation')

#Test on GT or decoded input
parser.add_argument("--input_type", default="i3d_transcript", help="select input type: [\"decoded\", \"gt\"]")
parser.add_argument("--runs", default=0, help="save runs")
