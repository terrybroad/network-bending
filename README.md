# Network Bending

Software reposity for code for paper "Network Bending: manipulating feature activations in GANs"

Code is based on this [StyleGAN2 pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by rosinality. You can refer to this codebase for training your own models, or converting models from the official tensorflow implementation. 

## Requirements

I have tested on: (CUSTOM OPERATORS ARE CURRENTLY NOT WORKING)

* PyTorch 1.4.0
* CUDA 10.1
* OpenCV - Refer to [this dockerfile](https://github.com/pytorch/extension-script/blob/master/Dockerfile) for installation of correct version
* Libtorch 1.4 (pre-C++11) [download here](https://pytorch.org/get-started/locally/)

### Build custom torchscript operators
We have built a number of torchscript operators using OpenCV and libtorch, you will have to have downloaded libtorch and installed the correct version of OpenCV for this to work. See requirements above or [refer to the tutorial for writing your own torchscript operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) for me details. 

To build the custom operators you can use the bash script accompanying with the path the your downloaded and unzipped libtorch code

> chmod +x ./build_custom_trasnforms.sh

> ./build_custom_transforms.sh /path/to/libtorch

If you are having issues with this you can link to the libtorch source in your Pytorch package installation folder: https://discuss.pytorch.org/t/segmentation-fault-when-loading-custom-operator/53301/8?u=tbroad

### Download StyleGAN2 Model

You can download the official StyleGAN2 FFHQ 1024 model converted to PyTorch format here: https://drive.google.com/drive/u/0/folders/1kxzAxJ9jrU6z9CPBJ8I87dXy-NJFG4zs

### Download Clustering Models

Link to clustering models

### Generate images

You can either generate images from random latents:

> python generate.py --ckpt /path/to/model.pt --size 1024 --pics 10 --config/example_transform_config.yaml

Or from a latent vector that you have projected into styleGAN space:

> python generate.py --ckpt /path/to/model.pt --size 1024 --pics 1 --latent /path/to/latent.pt --config config/example_transform_config.yaml

If you are using layers with random parameters you can generate multiple different samples from the same latent:

> python generate.py --ckpt /path/to/model.pt --size 1024 --pics 100 --latent /path/to/latent.pt --config config/example_transform_config.yaml

### Project images to latent spaces

> python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] IMAGE1 ...

### Transform Config

Explain transform config and parameters for each transformation layer

### Transform based on clusters

Explain two ways to do clustering

### Train your own clustering models

Download data: <link>

Train:

Or generate your own data:

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
