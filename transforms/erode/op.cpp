#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

torch::Tensor erode(torch::Tensor image, int64_t dilation_size) {
  bool useIPP = cv::ipp::useIPP();
  cv::ipp::setUseIPP(false);
  std::cout << "init" << std::endl;
  torch::Tensor i2 = image.to(torch::kCPU);
  std::cout << "clone" << std::endl;
  // cv::Mat image_mat(/*rows=*/i2.size(0),
  //                   /*cols=*/i2.size(1),
  //                   /*type=*/CV_32FC1,
  //                   /*data=*/i2.data<float>());
  cv::Mat image_mat(/*rows=*/i2.size(0),i2.size(1),CV_32FC1);
  std::cout << "image mat" << std::endl;
  std::memcpy((void*)i2.data_ptr(), image_mat.data, sizeof(float)*i2.numel());
  std::cout << "memcpy" << std::endl;
  int dilation_type = cv::MORPH_ELLIPSE;
  std::cout << "dilation type" << std::endl;
  cv::Mat output_mat;
  std::cout << "declare output mat" << std::endl;
  cv::Mat element = cv::getStructuringElement( dilation_type,
                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       cv::Point( dilation_size, dilation_size ) );
  std::cout << "element" << std::endl;
  cv::erode( image_mat, output_mat, element );
  std::cout << "erode" << std::endl;

  torch::Tensor output =
    torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{image.size(0), image.size(1)});
  std::cout << "tensor" << std::endl;
  return output.clone().to(at::kCUDA);
}

static auto registry =
  torch::RegisterOperators("my_ops::erode", &erode);