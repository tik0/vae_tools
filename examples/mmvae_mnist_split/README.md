# mmvae_mnist_split

This folder evaluates the MMVAE on a split MNIST data set.

## mmvae_mnist_split.ipynb

Simple demonstration on how to train on a split MNIST data set

## mmvae_mnist_split_hpsearch.ipynb

This script is executed in the shell whth the specific seed value:

```
#!/bin/bash
jupyter nbconvert --to python mmvae_mnist_split_hpsearch.ipynb;
CUDA_VISIBLE_DEVICES="" PYTHONPATH="/home/twbadmin/repositories/vae_tools" python3 -m mmvae_mnist_split_hpsearch
```

- <#index> denotes a specific hyperparameter configuration
- All files are stored wrt. to their seed in <dump_loc>/<#seed>
- Stores the hyper parameters an loss history in pandas data frame `history.h5`
    - Stores the list of loss values in `h_list_*`
    - Stores the final loss values in `h_*`
- Stores the networks ...
    - encoder mean networks: `enc_mean_<#index>_ab_<mask>.h5`
    - encoder logvar networks: `enc_logvar_<#index>_ab_<mask>.h5`
    - decoder logvar network `dec_<#index>_ab_<mask>.h5`

## mmvae_mnist_split_eval.ipynb

Read all the training results and evaluates them

## mmvae_mnist_split_eval_jsd.ipynb

Evaluate the Jensen-Shannon Divergence of all trained networks

Calculation of the JSD does not well behave with multiple threads wenn the number samples increase. I assume it is because of the high thread intercommunication.
However, reducing the thread number of numpy's openblas to 2 allows parallel evaluation of all seeds as follows:
```
#!/bin/bash
jupyter nbconvert --to python mmvae_mnist_split_eval_jsd.ipynb
OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""  PYTHONPATH="/home/twbadmin/repositories/vae_tools" nice -n 5 python3 -c "import mmvae_mnist_split_eval_jsd; mmvae_mnist_split_eval_jsd.run('0')" &
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""  PYTHONPATH="/home/twbadmin/repositories/vae_tools" nice -n 6 python3 -c "import mmvae_mnist_split_eval_jsd; mmvae_mnist_split_eval_jsd.run('1')" &
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""  PYTHONPATH="/home/twbadmin/repositories/vae_tools" nice -n 7 python3 -c "import mmvae_mnist_split_eval_jsd; mmvae_mnist_split_eval_jsd.run('2')" &
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""  PYTHONPATH="/home/twbadmin/repositories/vae_tools" nice -n 8 python3 -c "import mmvae_mnist_split_eval_jsd; mmvae_mnist_split_eval_jsd.run('3')" &
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""  PYTHONPATH="/home/twbadmin/repositories/vae_tools" nice -n 9 python3 -c "import mmvae_mnist_split_eval_jsd; mmvae_mnist_split_eval_jsd.run('4')" &
wait
```

