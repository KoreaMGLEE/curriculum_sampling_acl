import os
import sys
import torch
import random
import logging
import adapters
import argparse
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter

from adapters import BnConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# append project directory path to system path
sys.path.append('/mnt/user3/curriculum_sampling_acl/curriculum_sampling_acl/')

from configuration.args import args

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor, AutoConfig, T5ForConditionalGeneration
from models import GPT2LMHeadModel, LlamaForCausalLM

from utils import add_stdout_logger, build_dataloader, count_parameters, make_dir
from utils import load_data, normalize_answer, score

from peft_models import get_peft_model, LoraConfig, PromptTuningConfig
from datetime import datetime


# train code for regression model 
def main():
    add_stdout_logger()

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # load tokenizer
    if 't5' in args.model_name_or_path: 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif 'llama' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left')
    elif 'gpt' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    

    if 'llama' in args.model_name_or_path or 'gpt' in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        config.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id

    # load pre-trained model
    if 'llama' in args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, config=config)


    elif 'gpt' in args.model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    elif 't5' in args.model_name_or_path:
        if '11b' in args.model_name_or_path or '3b' in args.model_name_or_path:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config,
                                                               torch_dtype=torch.bfloat16)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)

    if args.method == "lora":
        peft_config = LoraConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False, r=args.reduction_factor, lora_alpha=8, lora_dropout=0.1)
    elif args.method == "prompt-tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_2_SEQ_LM", num_virtual_tokens=args.num_virtual_tokens)

    elif args.method == "adapter":
        adapter_config = BnConfig(mh_adapter=args.mh_head, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity="relu", original_ln_before=True)
        adapters.init(model)
        model.add_adapter("adapter", config=adapter_config)
        model.train_adapter(["adapter"])
        logging.info("trainable params: %s", count_parameters(model))

    if args.method is not None and args.method != "full":
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, relative_step=False, weight_decay=1e-5,
                          scale_parameter=False)

    for n, p in model.named_parameters():
        if p.requires_grad:
            if random.random() > 0.8:
                print(n, p.requires_grad)

    if args.do_train:
        # make save directory
        make_dir(args.output_dir)
        logging.info("Saving model to %s" % args.output_dir)
        
        # load train data
        train_features = load_data(args.task_name, 'train', args.model_name_or_path, tokenizer)

        if 'gpt' in args.model_name_or_path or 'llama' in args.model_name_or_path:
            train_dataloader = build_dataloader(train_features, args.train_bs, args.seed, model_name_or_path=args.model_name_or_path)
        elif 't5' in args.model_name_or_path:
            train_dataloader = build_dataloader(train_features, args.train_bs, args.seed, model_name_or_path=args.model_name_or_path)


        num_train_optimization_steps = int(len(train_features) / args.train_bs / args.gradient_accumulation_steps) * args.num_train_epochs
        logging.info(f"***** Running training ***** \n Num examples = {len(train_features)} \n Batch size = {args.train_bs} \n Number of training steps = {num_train_optimization_steps}")

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        model.to(device)

        tr_loss, global_step, total_steps, loss_ema, decay = 0, 0, 0, 0, 0.99

        # Create a directory with model name and current date
        current_time = datetime.now().strftime('%Y%m%d')
        log_dir = os.path.join(args.output_dir, f"{args.model_name_or_path.split('/')[-1]}_{current_time}_{args.reduction_factor}")

        print("="*100)
        logging.info(f"Saving logs to {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize SummaryWriter with the created directory
        tb_writer = SummaryWriter(log_dir=log_dir)

        # Save args configuration
        args_config_path = os.path.join(log_dir, 'args_config.txt')
        with open(args_config_path, 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        
        for e in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            model.train()
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)

                example_ids, input_ids, mask, segment_ids, label_ids = batch

                outputs = model(input_ids=input_ids, attention_mask=mask, token_type_ids=segment_ids,
                                                          labels=label_ids)                
                loss = outputs[0]
                total_steps += 1
                loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
                descript = "loss=%.4f" % (loss_ema / (1 - decay ** total_steps))
                pbar.set_description(descript, refresh=False)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                optimizer.step()
                global_step += 1
                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    tb_writer.add_scalar("loss", tr_loss / args.log_interval, global_step)
                    tr_loss = 0

            if e % 1 == 0:
                model.eval()
                eval_features, all_label_idsl, dataset_names = load_data(args.task_name, 'eval', args.model_name_or_path, tokenizer)
                acc_list = []
                for idx, eval_feature in enumerate(eval_features):
                    name = dataset_names[idx]
                    eval_dataloader = build_dataloader(eval_feature, args.eval_bs, args.seed, model_name_or_path=args.model_name_or_path)
                    iter_count = 0
                    predl, answerl = [], []
                    for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                                              desc="Evaluating",
                                                                                              ncols=100):
                        input_ids, input_mask = input_ids.to(device), input_mask.to(device)
                        
                        iter_count += 1
                        with torch.no_grad():
                            model_predictions = model.generate(input_ids=input_ids,
                                                               attention_mask=input_mask,
                                                               max_new_tokens=args.max_new_tokens, num_beams=1,
                                                               early_stopping=True, pad_token_id=tokenizer.eos_token_id)

                            predl.extend(tokenizer.batch_decode(model_predictions, skip_special_tokens=True))
                            
                            label_ids[label_ids == -100] = tokenizer.pad_token_id
                            answerl.extend(tokenizer.batch_decode(label_ids, skip_special_tokens=True))


                    
                    norm_predl = [normalize_answer(p) for p in predl]
                    norm_answerl = [normalize_answer(a) for a in answerl]

                    print("-"*100)
                    print(norm_predl[:10])
                    print(norm_answerl[:10])

                    if "hans" in name:
                        if 'gpt' in args.model_name_or_path or 'llama' in args.model_name_or_path:
                            norm_predl = ['entailment' if 'entailment' in p else 'not_entail' for p in norm_predl]
                            norm_answerl = ['entailment' if 'entailment' in a else 'not_entail' for a in norm_answerl]
                        else:
                            norm_predl = ['entailment' if p == 'entailment' else 'not_entailment' for p in norm_predl]
                            norm_answerl = ['entailment' if a == 'entailment' else 'not_entailment' for a in norm_answerl]

                    if 'mnli_train' in name:
                        result = {'acc': 0}
                    else:
                        acc_result = score('acc', norm_predl, norm_answerl, is_wsc=(args.task_name == 'wsc'),
                                           model=args.model_name_or_path)
                        result = {'acc': acc_result}
                    acc_list.append(result["acc"])


                    logging.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logging.info("%s  %s = %s", name, key, str(result[key]))

                output_all_eval_file = os.path.join(args.output_dir, f"{args.task_name}_eval_results.txt")
                with open(output_all_eval_file, "a") as all_writer:
                    all_writer.write(f"Eval results ({args.method}, {args.model_name_or_path}, {args.learning_rate}):\n")
                    all_writer.write("%s\n" % str(acc_list))
                tb_writer.close()

                model.train()


if __name__ == "__main__":
    main()