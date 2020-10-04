#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor resize(torch::Tensor image, double border_x, double border_y) {
  image = image.to(torch::kCPU);
  cv::Mat image_mat(image.size(0),
                    image.size(1),
                    CV_32FC1,
                    image.data_ptr<float>());
  std::cout << "image_in_size" << image.size(0) << std::endl;

  cv::Mat output_mat;

  cv::copyMakeBorder(image_mat, 
                     output_mat, 
                     double(image.size(1)) * (border_y/2.0), 
                     double(image.size(1)) * (border_y/2.0),
                     double(image.size(0)) * (border_x/2.0),
                     double(image.size(0)) * (border_x/2.0), 
                     cv::BORDER_REPLICATE);
  std::cout << "image_out_size" << output_mat.rows << std::endl;
  torch::Tensor output =
    torch::from_blob(output_mat.ptr<float>(),{output_mat.rows, output_mat.cols});
  return output.clone().to(torch::kCUDA);
}

static auto registry =
  torch::RegisterOperators("my_ops::resize", &resize);