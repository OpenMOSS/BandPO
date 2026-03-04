# 不要在ray的脚本中定义环境变量，需要在env的yaml文件写入
# 默认prompt_key=prompt，dapo原始数据集使用的就是prompt
# 但是dapo-processed使用的是source_prompt！！！！！！！！！可以考虑prompt_key=source_prompt
# 又但是aime使用的是默认prompt_key=prompt，最终只能modify dapo-processed数据集为prompt_key=prompt
# 数据集的data_source决定了rule-based reward返回type是dict还是float，但是整个训练中只能使用其中一种type（可能包含多种data_source），否则会导致/verl/trainer/ppo/ray_trainer.py中的assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}断言报错
# dapo按照gen_prompt_bsz切分数据集，通常要多个gen_prompt_bsz才能达到train_batch_size大小（被dynamic filting了很多）
# 这导致很多问题都被浪费了，因此需要更多轮epoch来充分使用数据集训练，把那些曾经太难的数据集全错的题开始逐渐做对
# 多个slurm节点跑独立的ray实验，需要在每个节点起ray node
# ray使用大量存储，设置额外的tmp的dir，ray子目录中每个session的时间代表GCS节点启动的时间
# export TMPDIR=/remote-home1/yli/tmp
# export TEMP=$TMPDIR
# export TMP=$TMPDIR
# 每个session都有一个working_dir_files在子目录runtime_resources中，working_dir_files里面的多个ray_pkg代表不同任务，
# 其中有一个working_dir存储着wandb文件
# 每个session都有一个logs，里面有多个job的logs，console输出就在对应job的job-driver-raysubmit_XXXXX.log文件中
# 数据集同名的键，其值类型要相同，否则pyarrow merge会报错。
# 测试集的data_source可以设置为"aime_xxx"来走math_dapo，又能区分开来
# val.n要大于1才能开启maj和best和worst
# 要是有KL项，比如GRPO，一定要：    
# actor_rollout_ref.actor.fsdp_config.param_offload=False \
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
# 否则会非常慢

# DAPO不支持重启！！！否则重启后和重启前的datasets对应的batch是无法匹配上的，假设gloable steps是35，重启前的batch用到了57/600，重启后用到了35/600。这是grpo的逻辑不是dapo的逻辑，这是一个bug！！！

# 不能直接使用DAPO的ray改为GRPO，因为会导致保留原来的dapo的token-level loss gradien

# grpo的naive reward manager在调用reward判分函数时，也会分别看是dict还是float，但是所有数据集要统一
# dapo是打-1和1，math和gsm8k是打0和1。
# GRPO的论文在附录把规则式（Rule）outcome写得很清楚：打了0和1，所以要把dapo的数据集source改为math
```python
if isinstance(score, dict):
    reward = score["score"]
    # Store the information including original reward
    for key, value in score.items():
        reward_extra_info[key].append(value)
else:
    reward = score
```
但是由于提取答案的方式不一样，要改prompt，索性直接修改naive的分数为0和1.