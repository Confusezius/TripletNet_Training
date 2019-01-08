## Small repo to train triplet-loss based networks.

This was implemented for the following datasets:
  * [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200.html)
  * [CARS196](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

To use this repo, simply download and extract the datasets to the folder of your choice `loadpath`.
Decide on a target folder to save network weights and checkpoints `savepath`.

Then run ```python train.py --source_path loadpath --save_path savepath --dataset <dataset_of_choice from cub200,cars196,celeba>```
