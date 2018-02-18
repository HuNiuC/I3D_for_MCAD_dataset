# I3D_for_MCAD_dataset
This repository contains the preprocessing for the new multi_view dataset MCAD  http://mmas.comp.nus.edu.sg/MCAD/MCAD.html
and train(fine-tune) the I3D model on the last Inception Model(Inc).

The orignial I3D model is reported in the paper "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" 
by Joao Carreira and Andrew Zisserman. The paper was posted on arXiv in May 2017, 
and will be published as a CVPR 2017 conference paper.

The orignial code released by the authors of "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" 
is there https://github.com/deepmind/kinetics-i3d

1. Run python save_data.py 
to select the center 224x224 image crop from the video, The provided .npy file has shape (64, 224, 224, 3) for RGB.
where 64 is the number of the model receptive field .

2. Run python train_new_rgb.py 
fine_tune the model

3. Run python extract_feature.py  --imagenet_pretrained true --eval_type rgb
to extract featuro of the I3D features on the fine-tune model.

