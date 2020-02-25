#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor scale(torch::Tensor image, double scale) {
  cv::Mat image_mat(/*rows=*/image.size(0),
                    /*cols=*/image.size(1),
                    /*type=*/CV_32FC1,
                    /*data=*/image.data<float>());

  cv::Mat output_mat;
  cv::Point2f centre(((float)image.size(0))/2.0,((float)image.size(1))/2.0);
  cv::Mat rotationMat = cv::getRotationMatrix2D(centre,0,scale);
  cv::warpAffine(image_mat, output_mat, rotationMat, image_mat.size());

  torch::Tensor output =
    torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{image.size(0), image.size(1)});
  return output.clone();
}

static auto registry =
  torch::RegisterOperators("my_ops::scale", &scale);