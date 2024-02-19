
This codebase contains the implementation of constrained RLHF. 

Currently, the code is a bit hacky, supporting only composite reward models formed from the combination of METEOR and intent matching rewards, but we plan to provide a more general implementation soon. 

The codebase is structured on the excellent RL4LMs repository: https://github.com/allenai/RL4LMs. 

Constrained approaches can be run via the command:
```
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_constrained_ppo.yml --log_to_wandb --seed=0 --experiment_name=<exp_name>
```

More documentation coming soon! In the meantime, please check out the documentation for [RL4LMs](https://github.com/allenai/RL4LMs) as a guide. If you find this code useful, please consider citing our paper. Thank you!

```
@misc{moskovitz2023confronting,
      title={Confronting Reward Model Overoptimization with Constrained RLHF}, 
      author={Ted Moskovitz and Aaditya K. Singh and DJ Strouse and Tuomas Sandholm and Ruslan Salakhutdinov and Anca D. Dragan and Stephen McAleer},
      year={2023},
      eprint={2310.04373},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

