#pragma once
#include <opencv2/opencv.hpp>

#include "piperun.h"

namespace pipeline::yolo_utils {

void rawin_pre(cv::Mat input, nndeploy::device::Tensor* input_tensor);
void norm_pre(cv::Mat input, nndeploy::device::Tensor* input_tensor);

template <typename PipeCtxT, pipeline::Platform P=pipeline::Platform::CURRENT_PLATFORM>
void yolo_pre_process(PipeCtxT& ctx) {
  if constexpr (P == pipeline::Platform::LinuxAarch64) {
    rawin_pre(ctx.input, &ctx.input_tensor);
  } else {
    norm_pre(ctx.input, &ctx.input_tensor);
  }
}



}  // namespace pipeline::yolo_utils