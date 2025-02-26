#pragma once
#include "nntaskflow/infer_task.h"
#include "nntaskflow/platform.inl"
#include "yolo-utils/yolo_pre.h"
#include "yolo-utils/yolo_nms.h"

struct YoloCtx{
  std::shared_ptr<nndeploy::inference::Inference> infer;
  cv::Mat input;
  nndeploy::device::Tensor input_tensor;
  //order come from output_tensors_ map, ...
  //zp;/scale order come from rknn order
  //will cause stuble bug
  std::vector<std::unique_ptr<nndeploy::device::Tensor>> output_tensors;
  std::vector<pipeline::yolo_utils::YoloBox> boxes;
  int nc = 80;
  int nkpts = 17;
};

