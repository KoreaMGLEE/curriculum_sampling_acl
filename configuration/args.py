import argparse

parser = argparse.ArgumentParser()

# model config
parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
parser.add_argument("--method", default=None, type=str)
parser.add_argument("--reduction_factor", default=1, type=int)

# training config 
parser.add_argument("--task_name", default='MNLI', type=str)
parser.add_argument("--train_bs", default=4, type=int)
parser.add_argument("--eval_bs", default=4, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--seed", default=32, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--n_processes", default=4, type=int)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--output_dir", default=None, type=str)
parser.add_argument("--num_train_epochs", default=3, type=int)
parser.add_argument("--no_cuda", action='store_true')

# inference config 
parser.add_argument("--max_new_tokens", default=5, type=int)
parser.add_argument("--do_eval", action='store_true')

# log config
parser.add_argument("--log_interval", default=1000, type=int)
args = parser.parse_args()