# Embodied Navigation with Auxiliary Task of Action Description Prediction

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Kondoh_Embodied_Navigation_with_Auxiliary_Task_of_Action_Description_Prediction_ICCV_2025_paper.html)


## Setup
1. setup python env
   ```
   conda create -n descrl python=3.9 cmake=3.14.0 -y
   conda activate descrl
   ```
3. setup [habitat-lab](https://github.com/facebookresearch/habitat-lab)
   ```
   cd habitat-lab
   python setup.py develop --all
   ```
4. setup habitat-sim ([reference](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source))
5. setup [soundspaces](https://github.com/facebookresearch/sound-spaces)
   ```
   cd sound-spaces
   pip install -e .
   ```
   * And also you need to download some datasets ([reference](https://github.com/haru425/myss/blob/main/sound-spaces/soundspaces/README.md))


## Tutorials

### Training
1. Pre-train ADGenerator
  1. edit `xgenerator/shs/train.sh`
  2. run the followings:
      ```
      cd xgenerator
      conda activate descrl
      bash ./shs/train.sh
      ```
2. Pre-train ADPredictor
   1. edit `sound-spaces/train.sh`
      * here, `MODEL` should be `savi-iprl-pre-offpolicy`
   2. run the followings:
      ```
      cd sound-spaces
      conda activate descrl
      bash ./train.sh
      ```
3. Jointly training
   1. run the followings:
      ```
      cd sound-spaces
      conda activate descrl
      bash ./train.sh
      ```

### Evaluation
Edit `sound-spaces/eval.sh` and run the followings:
```
cd sound-spaces
conda activate descrl
bash ./eval.sh
```

### Test
Edit `sound-spaces/test.sh` and run the followings:
```
cd sound-spaces
conda activate descrl
bash ./test.sh
```


## Citation

```
@inproceedings{kondoh2025descrl,
  title={Embodied Navigation with Auxiliary Task of Action Description Prediction},
  author={Kondoh, Haru and Kanezaki, Asako},
  booktitle={Proceedings of International Conference on Computer Vision (ICCV)},
  year={2025},
}
```
