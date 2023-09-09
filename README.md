# Multi-Goal Audio-Visual Navigation using Sound Direction Map

## ブランチ

### main
* メインのブランチ
* Multi-Goal Audio-Visual NavigationやSound Direction Mapを実装してきたブランチ
* soundspaces2を動かすことを想定している

### ss1
* コンペティションに参加するための実装がされているブランチ
* soundspaces1を動かすことを想定している


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
4. `qsub -g tga-aklab job.sh`を実行


### config等の変更方法

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