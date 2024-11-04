# Semantic Audio-Visual Embodied Navigation (saven) Model

This is the codebase for the ICLR 2022 submission titled "Knowledge-driven Scene Priors for Semantic Audio-Visual Embodied Navigation".
We forked and used [SoundSpaces](https://github.com/facebookresearch/sound-spaces) codebase to implement our model.

# Install

- Install [habitat-lab v0.1.7](https://github.com/facebookresearch/habitat-lab) and [habitat-sim v0.1.7](https://github.com/facebookresearch/habitat-sim)
- Install this repo by running the following command:
```
pip install -e .
```

# Dataset Download

- Create a folder named `data` at root location.
- Download [Matterport3D](https://niessner.github.io/Matterport), and place it at `./data/scene_datasets/mp3d`.
- Download pre-trained word vectors by [GloVe](https://nlp.stanford.edu/projects/glove/): https://nlp.stanford.edu/data/glove.840B.300d.zip. Unzip: `unzip glove.840B.300d.zip`, and place it at `.data/glove_data/pre_trained_word_vectors`

# Scripts

- Collect semantic metadata about Matterport3D, and compute object-to-object, object-to-region, region-to-region relationships, and compute region cluster. Results are saved in `data/metadata/`:
```
python scripts/create_metadata_graph_mp3d.py
```

- Save vision and semantic observations in `data/scene_observations_gt/mp3d/`:
```
python scripts/cache_observations_saven.py
```

- Filter images based on criteria to find images that are good for training vision model, and save the list of nodes (point and rotation) and its semantic labels for each scene in `data/metadata/mp3d_scene_valid_semantic_nodes.bin`:
```
python scripts/create_vision_dataset_saven.py
```

- Find word embeddings for objects and regions from pre-trained word vectors by GloVe, and compute adjacency matrix. Results are saved in `.data/glove_data/`:
```
Run this Jupyter Notebook: scripts/glove_embedding.ipynb
```

- Create dataset for training and testing the agent. Dataset is saved in `.data/datasets/semantic_avn/mp3d/v1/`:
```
cd scripts/dataset_code
python create_semantic_avn_dataset.py
```

# Training

- Pre-training vision model:
```
python ss_baselines/saven/pretraining/vision_model_trainer.py --run-type train
```

- Testing vision model:
```
python ss_baselines/saven/pretraining/vision_model_trainer.py --run-type test
```

- Pre-training audio model:
```
python ss_baselines/saven/pretraining/audio_model_trainer.py --run-type train
```

- Testing vision model:
```
python ss_baselines/saven/pretraining/audio_model_trainer.py --run-type test
```

- Pre-train the `saven` model (using the pre-trained vision and audio model). `Saven` is first trained with the external memory size of 1, which only uses the last observation:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven
```

- Evaluate the pre-training process. This will automatically run evaluation on the `val_seen-scenes_heard-sounds` data split for each of the checkpoints found in `data/models/saven/data`:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven --run-type eval EVAL.SPLIT val_seen-scenes_heard-sounds
```
Use the additional flag `--prev-ckpt-ind` to instead specify a starting checkpoint index `n` for the evaluation process, or to resume an evaluation process:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --prev-ckpt-ind n --model-dir data/models/saven --run-type eval EVAL.SPLIT val_seen-scenes_heard-sounds
```

- Once evaluation is complete, obtain the best checkpoint of the pre-training step and its corresponding metrics:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven --run-type eval --eval-best EVAL.SPLIT val_seen-scenes_heard-sounds
```

- Train the `saven` model using the best pre-trained checkpoint of pre-training it. Please update the `pretrained_weights` path in [saven.yaml](config/semantic_audionav/saven.yaml) with the best pre-trained checkpoint when finetuning:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml --model-dir data/models/saven
```

### Run the Seq2Seq baselines
This code includes the configuration files, policy and trainer to run six different Seq2Seq baselines:
- Train PointGoal (RGB + Depth + GPS):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/pointgoal/pointgoal.yaml --model-dir data/models/pointgoal
```
- Train ObjectGoal (RGB + Depth + GPS + Semantic Label):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/objectgoal/objectgoal.yaml --model-dir data/models/objectgoal
```
- Train AudioPointGoal (Audio + RGB + Depth + GPS):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-pointgoal/audio-pointgoal.yaml --model-dir data/models/audio-pointgoal
```
- Train AudioObjectGoal (Audio + RGB + Depth + GPS + Semantic Label):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-objectgoal/audio-objectgoal.yaml --model-dir data/models/audio-objectgoal
```
- Train AudioGoal (Audio + RGB + Depth):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-pointgoal/audiogoal.yaml --model-dir data/models/audiogoal
```
- Train AudioObjectGoal-NoGPS (Audio + RGB + Depth + Semantic Label):
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/simple_baselines/audio-objectgoal/audio-objectgoal_no-gps.yaml --model-dir data/models/audio-objectgoal_no-gps
```
In our paper we report the results obtained using `AudioGoal` and `AudioObjectGoal-NoGPS`.

To evaluate either of the baselines add the flag ```--run-type eval``` and specify the appropriate evaluation split, e.g., ```EVAL.SPLIT val_seen-scenes_heard-sounds```. 

### Run the random baselines
There are two random baselines `RandomAgentWithoutStop` and `RandomAgentWithStop`. The former is random baseline that uniformly samples one of three actions (FORWARD, LEFT, RIGHT) and executes stop when the radius distance is less than the specified success distance (1 meter): 
```
python ss_baselines/saven/run.py --run-type eval --exp-config ss_baselines/saven/config/random_agent_wo-stop.yaml
```
The latter samples one of four actions (FORWARD, LEFT, RIGHT, STOP) where STOP has a much lower probability of being selected:
```
python ss_baselines/saven/run.py --run-type eval --exp-config ss_baselines/saven/config/random_agent_w-stop.yaml
```
In our paper we report the results obtained using `RandomAgentWithoutStop`.

## Notes 
- Modify the parameter `NUM_UPDATES` in the configuration file according to the number of GPUs
