#include "yolo_pre.h"
namespace pipeline::yolo_utils {
void rawin_pre(cv::Mat input, nndeploy::device::Tensor* input_tensor) {
  cv::Mat matin;
  cv::resize(input, matin, cv::Size(640, 640));
  cv::cvtColor(matin, matin, cv::COLOR_BGR2RGB);
  nndeploy::device::Device* device = nndeploy::device::getDefaultHostDevice();
  nndeploy::device::TensorDesc desc;
  desc.data_type_ = nndeploy::base::dataTypeOf<uint8_t>();//!!
  desc.data_format_ = nndeploy::base::kDataFormatNHWC;
  desc.shape_ = {1, 640, 640, 3};
  input_tensor->create(device, desc, matin.data);
}

void norm_pre(cv::Mat input, nndeploy::device::Tensor* input_tensor) {
  cv::Mat div255;
  cv::resize(input, div255, cv::Size(640, 640));
  div255.convertTo(div255, CV_32FC1, 1.0 / 255.0);
  cv::cvtColor(div255, div255, cv::COLOR_BGR2RGB);
  nndeploy::device::Device* device = nndeploy::device::getDefaultHostDevice();
  nndeploy::device::TensorDesc desc;
  desc.data_format_ = nndeploy::base::kDataFormatNCHW;
  desc.shape_ = {1, 3, 640, 640};
  input_tensor->create(device, desc);//use create instead of reassign

  std::vector<cv::Mat> planes(3);
  float* dataptr = (float*)input_tensor->getData();
  for (size_t i = 0; i < 3; i++) {
    float* cur_ptr = dataptr + i * 640 * 640;
    planes[i] = cv::Mat(640, 640, CV_32FC1, cur_ptr);  // warn: hardcoded type !
  }
  cv::split(div255, planes);
  input_tensor->getDesc().print();

}
}  // namespace pipeline::yolo_utils