# DenseDepth-Torch

A simple PyTorch Implementation of  the "High Quality Monocular Depth Estimation via Transfer Learning" paper.

The paper can be read [here](https://arxiv.org/abs/1812.11941).

Official Implementation can be found [here](https://github.com/ialhashim/DenseDepth).

## Training

* The model was trained on the [ESPADA](https://ccc.inaoep.mx/~carranza/ESPADA/) dataset.



```shell
$ python train.py --epochs 31 --batch 4 --save ./checkpoints/ --device cuda
```

* The model was trained on Google Colab 30 epochs (~ 7/8 hours), it was trained periodically when Nvidia T4 or P100s were available. Training on a single 12 GB Tesla K80 takes too long. In contrast, the authors use a cluster of 4 Tesla K80s. In contrast, the authors train for 1 M epochs for 20 Hours.

* Train Loss at the end of the 20th epoch was  ~0.082.

## Usage

* *Step 1:* Clone the repository

```python
git clone https://github.com/AldrichCabrera/DenseDepth-Torch.git
```

* *Step 2:* Download ESPADA dataset or use your own dataset

* *Step 3:*  To train,

```
python train.py --epochs 31 --batch 4 --save ./checkpoints/ --device cuda
```

* *Step 4:* To test,

```
python test.py --checkpoint ./checkpoints/ckpt_5_7.pth --device cuda --data ./examples/
```

For help,

```
python train.py --help
```
