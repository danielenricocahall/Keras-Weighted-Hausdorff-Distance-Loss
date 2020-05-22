# Weighted Hausdorff Distance Loss #
In this repository, you'll find an implementation of the weighted Hausdorff Distance Loss, described here (https://arxiv.org/abs/1806.07564). A majority of the work was just porting their PyTorch implementation (https://github.com/HaipengXiong/weighted-hausdorff-loss). I figured some researchers/practitioners that are doing object detection/localization may find this useful!

# Setup

`pipenv install .` should configure a python environment and install all necessary dependencies in the environment. 

# Testing

Some tests verifying basic components of the loss function have been incorporated. Run `python -m pytest` in the repo to execute them.
## TODO ## 
Add an example script.
