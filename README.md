

[toc]

# MAI-VSR-Diggers

Team "Diggers" winner solution to Mobile AI 2021 Real-Time Video Super-Resolution Challenge

official report paper: https://arxiv.org/abs/2105.08826

## Pipeline

![](https://www.hualigs.cn/image/60578e1618d5f.jpg)

[![6IemqS.png](https://z3.ax1x.com/2021/03/22/6IemqS.png)](https://imgtu.com/i/6IemqS)

# usage

## install

* Linux machine (you do not need to care about cuda version, only need NVIDIA graphics driver version greater than 418)
* python 3.7  virtual env
* `pip install megengine -f https://megengine.org.cn/whl/mge.html`
* `pip install -r requirements.txt `

## dataset preparation (REDS)

* link:    https://seungjunnah.github.io/Datasets/reds.html
* after unzip it ,you need to **merge** the training and validation dataset(like mmediting), thus total 270(240+30) clip, and remaining 30 clip for test.
* after merging,  your dir should like this:
  * train
    * train_sharp
      * 000
      * ...
      * 240 (the first validation clip, thus clip 000 of validation)
      * ...
      * 269
    * train_sharp_bicubic
      * X4
        * 000
        * ...
        * 269
  * test
    * test_sharp_bicubic
      * X4
        * 000
        * ...
        * 269

## Training

* find the config file:  `configs/restorers/BasicVSR/mai.py `
* change the first few lines according your situation:

<img src="https://www.hualigs.cn/image/60579011b098b.jpg" style="zoom: 50%;" />

* start to run:

```bash
cd xxx/MAI-VSR-Diggers
python tools/train.py configs/restorers/BasicVSR/mai.py --gpuids 0,1,2,3 -d
```

> support multi gpus training, change to yours, e.g.  --gpuids 0     --gpuids 0,2       etc...

you can find output information and checkpoints in `.workdirs/...`

## Testing  (now only support REDS dataset)

### our checkpoint

use our trained model (**generator_module.mge**), already inside this repo:  `./ckpt/epoch_62`   which is only  **92kb**

**it has been trained 62 epochs on 240 clips, it's PSNR on validation dataset(3000 frames) is 27.98**

### test on valid dataset

find the config file:  `configs/restorers/BasicVSR/mai_test_valid.py` 

change first lines for your situation, **actually you only need to fix the dataroot**

```python
dataroot = "/path2yours/REDS/train/train_sharp_bicubic"
load_path = './ckpt/epoch_62'
exp_name = 'mai_test_for_validation'
eval_part = tuple(map(str, range(240, 270)))
```

and then , run  it:

```bash
cd xxx/MAI-VSR-Diggers
python  tools/test.py  configs/restorers/BasicVSR/mai_test_valid.py --gpuids 0 -d
```

you can find the results in `./workdirs/...`

### test on test dataset

find the config file:  `configs/restorers/BasicVSR/mai_test_test.py` 

change first lines for your situation, **actually you only need to fix the dataroot**

```python
dataroot = "/path2yours/REDS/test/test_sharp_bicubic"
load_path = './ckpt/epoch_62'
exp_name = 'mai_test_for_test'
eval_part = None
```

and then , run  it:

```bash
python  tools/test.py  configs/restorers/BasicVSR/mai_test_test.py --gpuids 0 -d
```

you can find the results in `./workdirs/...`

> notice: only support one gpu config for gpuids now

## Results on testset

* all output frames of test dataset produced by our model can be found here:  (3000 frames, trained only on 240 training clips): 

https://drive.google.com/file/d/1R0DDHmV8jZW_iYJQZksO2RWkZTrAYYPi/view?usp=sharing

## get the tflite model

### Overall pipeline thinking

* train the model by [megengine](https://megengine.org.cn/) framework(something like pytorch, tensorflow....)
* definite same model by tensorflow (same size, same deal pipeline...)
* load xxx.mge -> numpy.ndarray -> tf.keras.Model
* convert the tf.keras.Model to tflite using tensorflow

### one line to get .tflite

#### model.tflite

```
cd xxx/tflite/
python main.py  --mgepath  /xxxxxx/ckpt/epoch_62/generator_module.mge
```

#### model_none.tflite

```bash
cd xxx/tflite/
python main.py  --mgepath  /xxxxxx/ckpt/epoch_62/generator_module.mge  -n
```

> notice that to use **absolute** path 

you will get tflite files in the dir `xxx/tflite/xxx`

and we have supported our pre-built  `model.tflite`  and `model_none.tflite`  in `ckpt` dir

## testing on custum data using tflite
you can refer to https://github.com/Feynman1999/MAI-VSR-Diggers/issues/2.
