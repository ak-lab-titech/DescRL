# Multi-Goal Audio-Visual Navigation using Sound Direction Map

## ブランチ

### main
* メインのブランチ
* Multi-Goal Audio-Visual NavigationやSound Direction Mapを実装してきたブランチ
* soundspaces2でaudio-visual navigationを動かすことを想定している

### ss1
* コンペティションに参加するための実装がされているブランチ
* soundspaces1でaudio-visual navigationを動かすことを想定している

### iprl-discrete
* 離散環境下で、インストラクション予測を補助タスクにした強化学習を行うためのブランチ
* soundspaces1でsemantic audio-visual navigationを動かすことを想定している


## 環境構築

1. habitat-labのセットアップ
    ```
    cd habitat-lab
    python setup.py develop --all
    ```
2. habitat-simのセットアップ
    * [参考](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source)
    * 卒論で用いたのはコミットID: 80f8e31140eaf50fe6c5ab488525ae1bdf250bd9
3. `anaconda3/envs/av-nav/lib/python3.9/site-packages/habitat_sim-0.2.2-py3.9-linux-x86_64.egg/habitat_sim`の一部変更
    * `habitat_sim/simulator.py`をこのリポジトリの`my_habitat_sim/simulator.py`に移し替える
4. soundspacesのセットアップ
    ```
    cd sound-spaces
    pip install -e .
    ```
5. データセットのダウンロード ([参考](https://github.com/haru425/myss/blob/main/sound-spaces/soundspaces/README.md))
    1. 以下を実行。SoundSpaces2を使う場合は`binaural_rirs.tar`は必要ない。
        ```
        mkdir data
        cd data
        wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz
        wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz
        ```
    2. Replicaを`./data/scene_datasets`にダウンロード ([参考](https://github.com/facebookresearch/Replica-Dataset#download-on-mac-os-and-linux))



## 実行方法

### 基本的な実行方法 (TSUBAME)
1. `./sound-spaces`に移動
2. `job.sh`の変更
    * 目的に応じて`CMD`を変更する
        * `multi-gpu-train`: マルチGPUで訓練
        * `single-gpu-train`: 一つのGPUで訓練
        * `test`: 学習したモデルのテスト
        * `video`: エージェントが室内環境で動いているvideoを生成する
        * `make_dataset`: datasetを作る
        * `plot_tb_data`: tensorboardのデータをpythonのpltでplotする
    * ノードの種類、実行時間、outputファイル名を変更する
3. 実行するシェルスクリプトの中身(train.sh, test.sh, video.shなど)を必要に応じて変更する
    * videoとtestの場合はファイル頭部の変数と、AV-NavなのかSAViなのか
    * trainの場合はモデルを保存するディレクトリ名と、AV-NavなのかSAVi 1stなのかSAVi 2ndなのか
    * w/ IPRLで学習したものをevalとtestする場合でも、SNSORSからGENERATED_INSTRUCTIONを外した方が高速化されるので外した方が良い。evalとtestでは使わないので
4. `qsub -g tga-aklab job.sh`を実行


### config等の変更方法

#### IPRLの使用
1. `myss/sound-spaces/ss_baselines/savi/config/semantic_audionav/savi_pretraining.yaml`(savi-1st)または同ディレクトリの`savi.yaml`(savi-2nd)の`RL.PPO.INSTRUCTION_PREDICTOR.use_iprl`をTrueにする。また、`use_instruction_predictor`もTrueにする必要がある。
2. `RL.PPO.INSTRUCTION_PREDICTOR`内を必要に応じて変更
3. `myss/sound-spaces/configs/semantic_audionav/savi/mp3d/semantic_audiogoal.yaml`の`TASK.SENSORS`に`'GENERATED_INSTRUCTION'`を追加
4. `TASK.GENERATED_INSTRUCTION`を必要に応じて変更。


#### IPRLの事前学習

##### 事前学習用のデータを作成する
GeneratedInstructionで時間がかかりすぎるので、事前に作成しておくと高速化ができる。
1. DATASET.SPLITで、どのデータセットにおいて生成したいのか選択
    * 1データあたり0.4sくらいかかるので、膨大すぎる場合は事前にsplitして並列でやるのが良い

episodeをsimulationして画像とかまで保存する。diskにあらかじめ保存しておくことでoff-policy事前学習の高速化を期待
1. DATASET.SPLITで、どのデータセットにおいて生成したいのか選択
    * train_w_instruction, val_w_instruction, test_w_instructionで指定する。通常のtrain, val, testで指定すると、episodeデータのindexと合わなくなるので。


##### 事前学習
on-policyで行う場合
1. `myss/sound-spaces/configs/semantic_audionav/savi/mp3d/semantic_audiogoal.yaml`の`TASK.SENSORS`に`'ORACLE_ACTION_SENSOR'`を追加
2. SENSORSの`'GENERATED_INSRUCTION'`を消して`'ORACLE_ACTION_GENERATED_INSTRUCTION'`を追加

off-policyで行う場合
1. SENSORSに`ORACLE_ACTION_SENSOR`も`GENERATED_INSTRUCTION`も`ORACLE_ACTION_GENERATED_INSTRUCTION`も不要
    * (そもそも呼ばれないので消す必要もない)
2. SPLITを、instructionが含まれているものを選択するようにする
    * e.g. `train_w_instruction`
3. batch_size (PPO.num_steps)は64くらいまでおとさないとmemoryが足りないのでおとす
    * ついでにTFのメモリーサイズも64まで落としておいた方が良いかも
4. multi gpuで学習させたい場合も、job.shのCMDは`single-gpu-train`にする

#### 事前学習ずみをSAViで使う
1. IPRLの事前学習後にこの重みを利用したい場合は、`myss/sound-spaces/ss_baselines/savi/config/semantic_audionav/savi.yaml`のSCENE_MEMORY_TRANSFORMERの`use_pretrained`をTrueにして、`pretrained_path`を指定する。

#### Semantic画像を用いたXGeneratorの使用
1. GENERATED_INSTRUCTION.XGENERATOR_PATHとGENERATED_INSTRUCTION.XGENERATOR_CKPTをSemantic画像対応のXGeneratorに変更
2. SENSORSに`"SEMANTIC_SENSOR"`を追加

#### K-SAVEN
[Knowledge-driven Scene Priors for Semantic Audio-Visual Embodied Navigation
](https://arxiv.org/abs/2212.11345)

audioとvisualの事前学習の際は、Multi-GPUでやる場合も`CMD=single-gpu-train`にする

savi1stとsavi2ndの学習時
1. `TASK.SENSOR` の `CATEGORY_BELIEF` を `KSAVEN_CATEGORY_BELIEF`にする
2. 2ndの場合は、`pretrained_weights`を要変更


#### 音を永続的にする場合
1. `SIMULATOR.AUDIO.EVERLASTING`をTrueにする
2. train.shで、PRETRAINED_MODEL_NAMEとCKPT_NUMを永続的なもので学習したやつにする

#### ゴール数の変更
1. configの`SIMULATOR.AUDIO.NUM`を変更する
2. configの`DATASET.SPLIT`を変更する
    * 1-goalなら`train_telephone`、2-goalなら`train_telephone_bell`、3-goalなら`train_telephone_bell_birds1`

#### w/o SDM
デフォルトではSDMを用いるようになっている。以下は用いないようするための変更方法
1. configの`SIMULATOR.DIRECT_MAP_SIZE`を8ではなく`null`にする
2. configの`TASK.SENSORS`から`DIRECT_MAP`を外す

#### テストをするための変更点
1. テストの場合は、`EVAL.SPLIT`をtest用のデータセットに変更する。1-goalなら`test_telephone`、2-goalなら`test_telephone_bell`、3-goalなら`test_telephone_bell_birds1`
    * ただし、split testを行いたい場合は、これらの語尾に`_0`, `_1`, ..., `_4`をつけて5回に分ける。
2. split testではない場合は、`TEST_EPISODE_COUNT`を1000にするように。
3. `DATASET.SOUND_TYPE`は`multi_test`にしなければ未聴音にならない

#### タイミングのテストする際の注意点
* タイミングの変更はconfigで行わない
* `myss/sound-spaces/soundspaces/continuous_simulator.py`の494行目の`timing`を変更する

#### 同一音と不同音のテストする際の注意点
* こちらもconfigで行わない
* `myss/sound-spaces/soundspaces/datasets/audionav_dataset.py`の231~239行目のコメントアウトによって変更する

#### videoを作成するための変更点
1. `VIDEO_OPTION`に`"disk"`を追加する
2. av-navの場合は`TASK.SENSORS`に`"POINTGOAL_WITH_GPS_COMPASS_SENSOR","AUDIOGOAL_SENSOR"`を追記する
3. top down mapも作成したい場合は`VISUALIZATION_OPTION`に`"top_down_map"`を追記する