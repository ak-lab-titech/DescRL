# XGenerator

事前学習である、説明生成器の学習や評価を行うディレクトリ


## モデル
* speaker
    * [Speaker-Follower Models for Vision-and-Language Navigation](https://arxiv.org/abs/1806.02724)のSpeakerを[VLNCE](https://arxiv.org/abs/2004.02857)のR2Rで再現したもの
    * LSTMを元にしたSeq2Seqモデル
    * NeurIPS2022の[AVLEN](https://arxiv.org/abs/2210.07940)の論文でも用いられている
* transformer-speaker
    * speakerではうまくいかなかったので、transformerで実装したもの
* timesformer-gpt2
    * video captioningの事前学習済みモデル
    * videoのencoderとしてtimesformerを、文章生成のdecoderとしてGPT2を用いたもの
    * [参考](https://huggingface.co/Neleac/timesformer-gpt2-video-captioning)


## 実行方法

### Train
モデルの学習を行う。

1. `job.sh`の編集
    1. `CMD`を`train`に設定
    2. 学習したいモデルに応じて`MODEL`を編集
    3. `MODEL_NAME`でモデルの名前（保存するディレクトリの名前）を決める
2. 以下を実行
    ```
    cd myss/xgenerator
    qsub -g tga-aklab job.sh
    ```

### Eval
学習させたモデルの定性評価を行う。

1. `job.sh`の編集
    1. `CMD`を`eval`に設定
    2. 評価したいモデルに合わせて`MODEL`を編集
    3. `MODEL_NAME`で評価したいモデルの名前を指定
    4. `EVAL_CKPT_NUM`で、評価したいモデルのチェックポイントを指定
    5. `EVAL_NUM`で、何回定性評価を行うか指定
2. 以下を実行
    ```
    cd myss/xgenerator
    qsub -g tga-aklab job.sh
    ```


## Tensorboard
```
cd myss/xgenerator
export OPENBLAS_NUM_THREADS=1
tensorboard --logdir ./data/models/model-name/tb
```
