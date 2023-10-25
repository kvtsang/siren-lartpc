# siren-lartpc
Implementation of sinusoidal representation network (`siren`) for modeling the transport of optical photon signals in Liquid Argon Time Projection Chambers (LArTPCs).

## Installation
This repository requires `photonlib` package that you can download from [here](https://github.com/drinkingkazu/photonlib).
Install the `photonlib` following the instructions on its webpage.

Then `git clone` this repository and:
```
pip install .
```

## Siren for modeling optical photon transport
In the simplest term, the optical photon transport can be modeled by a function that calculates the probability for the detector $d$ to observe a photon produced at a position $\vec{r}$.
A naive form of implementation is to only estimate the mean probability in a deterministic manner (i.e. ignore stochasticity over a distribution), namely $f:R^N\mapsto R$. 

Traditionally, LArTPC experiments have employed [Photon Library (here)](https://github.com/drinkingkazu/photonlib) for this modeling.
We have shown that `siren`, a neural network designed to model a continuous field in space (and learns accurate gradients), brings significant advantages to replace Photon Library.
Read the original paper to learn about `siren` [here](https://www.vincentsitzmann.com/siren/).
The study is shown in [this paper](https://arxiv.org/pdf/2211.01505.pdf) where known issues (in particular, scalability) for Photon Library are also discussed.
This repository implements siren for LArTPC optical photon transport including scripts to run optimization of siren.

## Training siren
You need two items:
* a data file for Photon Library
  * You can train `siren` given a Photon Library data file. Follow the instructions [here](https://github.com/drinkingkazu/photonlib) and download the photon library file.   
* a configuration file to run the training script
  * You can prepare by yourself or use an example provided in this repository at `slar/configs` directory. 

Example configuration files can be found from the terminal:
```
python3 -c "from slar.utils import get_config;print(get_config('icarus_train'))"
```


