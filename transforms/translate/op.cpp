#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor translate(torch::Tensor image, double translate_x, double translate_y) {
  cv::Mat image_mat(/*rows=*/image.size(0),
                    /*cols=*/image.size(1),
                    /*type=*/CV_32FC1,
                    /*data=*/image.data<float>());

  cv::Mat output_mat;
  cv::Mat translationMat = (cv::Mat) cv::Matx23f(1,0,translate_x*(float)image.size(0),0,1,translate_y*(float)image.size(1));
  cv::warpAffine(image_mat, output_mat, translationMat, image_mat.size());

  torch::Tensor output =
    torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{image.size(0), image.size(1)});
  return output.clone();
}

static auto registry =
  torch::RegisterOperators("my_ops::translate", &translate);