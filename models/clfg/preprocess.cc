#include "preprocess.h"

#include "nndeploy/device/device.h"
#include "nndeploy/base/type.h"
#include "nndeploy/preprocess/opencv_convert.h"

namespace models {
namespace clfg {

PreprocessNode::PreprocessNode(const std::string& name, dag::Edge* input,
                               dag::Edge* output)
    : dag::Node(name, input, output), input_(input), output_(output) {}

PreprocessNode::~PreprocessNode() {}

base::Status PreprocessNode::run() {
  // 从输入边获取数据
  cv::Mat* src = inputs_[0]->getCvMat(this);
  if (!src) {
    return base::kStatusCodeErrorUnknown;
  }

  // 裁剪图像的中间部分
  cv::Mat cropped = (*src)(cv::Rect(80, 80, src->cols - 160, src->rows - 160));

  // 缩放到 160x160
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(160, 160));

  // 归一化处理
  resized.convertTo(resized, CV_32FC1, 1.0 / 255.0);
  resized -= 0.5;
  resized /= 0.5;

  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_format_ = base::kDataFormatNCHW;
  desc.shape_ = {1, 1, 160, 160};
  device::Tensor* dst = outputs_[0]->create(device, desc, 0);
  preprocess::OpenCvConvert::convertToTensor(resized, dst, false, nullptr,
                                             nullptr, nullptr);
  outputs_[0]->notifyWritten(dst);
  return base::kStatusCodeOk;
}

}  // namespace clfg
}  // namespace models