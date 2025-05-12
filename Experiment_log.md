# Experiment Log

#### Dataset Preparation
- DAPO-17k. The dataset in https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k is not equipped with the default prompt of Qwen2.5 series. Moreover, there are actually 17k prompts (that is why it is called DAPO-17k). Each prompt is repeated 100 times so the data size is 1.79M. So we do the following edits:
  * We use the prompts in ./examples/data-preprocess/math_dataset.py. 
  * For base models, we ignore the chat template. 
  * We deduplicate the dataset, and sample 1000 of them as the test set. The remaining 16398 is the training set.

#### Initial Test to Dataset
- Qwen2.5-1.5B
    * DAPO-17k-test-1000: pass@1 = 0.05.
    * pass@32: {'zeros': 0.54, 'ones': 0.0, '0.0-0.1': 0.299, '0.1-0.2': 0.091, '0.2-0.3': 0.042, '0.3-0.4': 0.013, '0.4-0.5': 0.01, '0.5-0.6': 0.004, '0.6-0.7': 0.001, '0.7-0.8': 0.0, '0.8-0.9': 0.0, '0.9-1.0': 0.0} 
- Qwen2.5-Math-1.5B (with chat template)
    * DAPO-17k-test-1000: pass@1 = 0.143
    * pass@32: {'zeros': 0.34, 'ones': 0.0, '0.0-0.1': 0.26, '0.1-0.2': 0.132, '0.2-0.3': 0.086, '0.3-0.4': 0.06, '0.4-0.5': 0.045, '0.5-0.6': 0.034, '0.6-0.7': 0.024, '0.7-0.8': 0.008, '0.8-0.9': 0.005, '0.9-1.0': 0.0}

#### Algorithm Design for RLOO
- For each RL step, we aim to have multiple prompts with N responses where there is at least one correct response and one incorrect response. When N is large this cost is huge. 
- So we use a smaller n and sample n responses for each prompt within a large batch. Ideally, we need n << N. If the pass rate of these n responseis noot 0 or 100, then we keep this prompt and sample n_continue = N-n responses. Otherwise, we discard this prompt.
- This is a very simple curriculum. Ideally, this specially works for two cases:
  * There is a huge dataset and doing RL on the whole dataset is infeasible based on the compute resource we have.
  * The distribution of the pass rate of all prompts is heavy-tailed. So most prompts have pass rates of 0 r 100 (or close to 0 or 100), and sampling n responses for these prompts is a waste.

#### Hyperparameters
```
curriculum.enable: whether we enable the curriculum.

data.gen_batch_size: how many prompts we try to generate n responses for each step (this is the generation batch size for the first generation stage).

data.train_batch_size: how many qualified prompts we need for each RL step. This is the batch size for the second generation stage. This is the number of actual  prompts  that we use to train the RL model in one RL step.

actor_rollout_ref.rollout.n: n.

actor_rollout_ref.rollout.n_continue: N-n.

max_num_gen_batches: the maximum number of generation batches we try to generate in the first stage.  If after this number of generation batches, we still don't have enough qualified prompts, we will raise an error.

```

#### Implementation
- The naive implementation will call 2 vLLM instances. However, this is not efficient. So we call 1 vLLM instance and use the `generate_sequences()` function to generate the responses.
- This is done by combining the second generation stage in this step with the first generation stage in the next step when calling the `generate_sequences()` function.

#### Detailed Edits to the Codebase
- We add `data.use_chat_template` to denote whether we use the chat template for the base model. This needs to edit the `RLHFDataset.__getitem__` function and the truncation part.
- Since we do not use the DAPO's prompt, we need to change the reward function. This can be doneby editing the function in `verl/utils/reward_score/__init__.py`. We use the `math_verify.compute_score()` function to compute the score.
- Modify the `./recipe/dapo/dapo_ray_trainer.fit()` function. Pay attention to the `ProtoData.union()` function because it is used to merge the prompts and the responses. To write the main algorithm, you also need to add `DataProto.truncate()` and `DataProto.interleave_by_uid()` functions in `verl/protocol.py`. The interleave function is to make the response in both stages (for the same prompt) are in the contiguous space in the batch so that we can easily compute the advantage and do the RL training.
- If you are using an old version of tensordict (say, 0.1.2), you should remove the multi-modal part in the `verl/workers/actor/dp_actor.py` file.
- We implement the `recipe/dapo/src/main_fast_dapo.py` and `recipe/dapo/src/fast_dapo_ray_trainer.py` to implement the fast DAPO algorithm.
- We implement the `verl/trainer/ppo/data_controller.py` to control the data for the PPO training. In `DataController`, we maintain several `DataProto` instances to store the prompts and responses for the first generation stage, the second generation stage, and the data ready for trainingrespectively.
  * `DataController.prompts_for_first_generation_phase`: the prompts for the first generation stage. When we call `add_new_prompts_first_phase()`, we add the new prompts to this instance.
  * `DataController.prompts_for_second_generation_phase`: the prompts for the second generation stage. 
  * `DataController.prompts_for_training`: the prompts for the training stage. `DataController.num_prompts_for_training` is the number of prompts in this instance. The function `is_ready_for_training()` is used to check whether we have enough prompts for training. This function returns True if the number of prompts in `DataController.prompts_for_training` is no less than `self.train_batch_size` (the `config.data.train_batch_size` in fast_dapo_ray_trainer.py).
  * After calling `self.actor_wg.generate_sequences()`, we call `DataController.update_prompts()` to sync the newly generated responses. Data flow: in the generation, there are two types of prompts: one with `self.config.actor_rollout_ref.rollout.n` responses and the other with `self.config.actor_rollout_ref.rollout.n_continue` responses. For the prompts that finished the first generation phase, we add the responses (together with the prompts and rewards) to `DataController.prompts_for_training`. We then add the qualified prompts to `DataController.prompts_for_second_generation_phase`. For the prompts that finished the second generation phase, we add the prompts and responses to `DataController.prompts_for_training`.
  * When it is ready for training, we call `DataController.get_training_data()` to get the data for training. This returns `self.config.data.train_batch_size` prompts with `self.config.actor_rollout_ref.rollout.n + self.config.actor_rollout_ref.rollout.n_continue` responses each.
- The entire loop for `RayFastDAPOTrainer.fit()` is as follows:
```
initialize the self.datacontroller
global_steps = 0

for every step:
    if not is_ready_for_training:

        batch = get a new batch from the training dataloader (ready for the first generation phase).

        self.datacontroller.add_new_prompts_first_generation_phase(batch).
        # Update the data controller: 

        batch = self.datacontroller.get_generation_inputs().
        # Get a new batch from the data controller. This includes the prompts for the first and second generation phase.

        gen_batch = construct the generation batch from batch

        gen_batch_output = self.actor_wg.generate_sequences(gen_batch)
        # Generate outputs

        # Call the reward function and compute the pass rates for the prompts in the gen_batch_output.

        self.datacontroller.update_prompts() # Update the data controller.

        continue

    elif is_ready_for_training:
        batch = self.datacontroller.get_training_data() # Get a batch of training data.

        # Compute the reference policy for the batch.

        # Compute the advantage and the loss.

        # Update the actors (and critics if necessary).

        # Log the metrics.

        # Update the data controller.

        global_steps += 1
        # The global steps is the actual number of RL steps.

        continue
```
- We save the metrics to a json file as a stream. This is done by `json.dumps()` and `jf.write()`. We log the total training time and the global steps.

#### Experiments and Baselines
- Models: Qwen2.5-1.5B. 
- Training set: DAPO-17k-train-16398, NuminaMath, OpenReasoningMath.
- Validation set: DAPO-1000 (a held-out set from DAPO-17k), AIME2024, AIME2025, AMC2023, MATH500.
- Baselines: Vanilla RLOO, Vanilla GRPO, Vanilla DAPO.