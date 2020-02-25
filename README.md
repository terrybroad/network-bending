# Network Bending

Software reposity for code for paper "Network Bending: manipulating feature activations in GANs"

Code is based on this [StyleGAN2 pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by rosinality.

## Requirements

I have tested on:

* PyTorch 1.3.1
* CUDA 10.1/10.2
* OpenCV

### Project images to latent spaces

> python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
