CUDA_VISIBLE_DEVICES=1 python ../../src/train.py --model_name_or_path openai-community/gpt2-medium \
 --do_train\
  --do_eval\
  --method lora \
   --output_dir ../../log/fever/gpt2-medium \
   --task_name FEVER\
   --num_train_epochs 5\
   --learning_rate 1e-5\
   --reduction_factor 32\
   --train_bs 4\

CUDA_VISIBLE_DEVICES=1 python ../../src/train.py --model_name_or_path openai-community/gpt2-medium \
 --do_train\
  --do_eval\
  --method lora \
   --output_dir ../../log/fever/gpt2-medium \
   --task_name FEVER\
   --num_train_epochs 5\
   --learning_rate 1e-4\
   --reduction_factor 32\
   --train_bs 4\
   --gradient_accumulation_steps 2\

CUDA_VISIBLE_DEVICES=1 python ../../src/train.py --model_name_or_path openai-community/gpt2-medium \
 --do_train\
  --do_eval\
  --method lora \
   --output_dir ../../log/fever/gpt2-medium \
   --task_name FEVER\
   --num_train_epochs 5\
   --learning_rate 1e-3\
   --reduction_factor 32\
   --train_bs 4\
   --gradient_accumulation_steps 2\

