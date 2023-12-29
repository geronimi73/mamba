export CUDA_VISIBLE_DEVICES=0
python3 lm_harness_eval.py --model mamba --model_args pretrained=outputs/52556da1-77a7-4c09-baa0-f964cd33cce5/epoch_2/step_4946 --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande --device cuda --batch_size 512
