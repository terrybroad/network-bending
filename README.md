# Network Bending

Software reposity for code for paper "Network Bending: manipulating feature activations in GANs"

Code is based on this [StyleGAN2 pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by rosinality. You can refer to this codebase for training your own models, or converting models from the official tensorflow implementation. 

## Requirements

I have tested on:

* PyTorch 1.4.0
* CUDA 10.1
* OpenCV - Refer to [this dockerfile](https://github.com/pytorch/extension-script/blob/master/Dockerfile) for installation of correct version
* Libtorch 1.4 (pre-C++11) [download here](https://pytorch.org/get-started/locally/)

### Build custom torchscript operators
We have built a number of torchscript operators using OpenCV and libtorch, you will have to have downloaded libtorch and installed the correct version of OpenCV for this to work. See requirements above or [refer to the tutorial for writing your own torchscript operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) for me details. 

To build the custom operators you can use the bash script accompanying with the path the your downloaded and unzipped libtorch code

> chmod +x ./build_custom_trasnforms.sh

> ./build_custom_transforms.sh /path/to/libtorch

### Project images to latent spaces

> python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
