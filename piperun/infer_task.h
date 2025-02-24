#pragma once

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference.h"
#include "taskflow/taskflow.hpp"  // Taskflow is header-only

// struct PipeContext {
//   nndeploy::inference::Inference* infer;
//   cv::Mat input;
//   nndeploy::device::Tensor input_tensor;
//   std::vector<nndeploy::device::Tensor> output_tensors;
//   tf::Task pre_task;
//   tf::Task infer_task;
//   tf::Task post_task;
// };

inline nndeploy::inference::Inference* create_inference(
    nndeploy::base::InferenceType type, 
    std::function<void(nndeploy::base::Param*)> custom_param = nullptr) {
  auto infer = nndeploy::inference::createInference(type);
  auto param = infer->getParam();
  if (custom_param != nullptr) {
    custom_param(param);
  }
  infer->init();
  return infer;
}

template <typename PipeCtxT>
tf::Task create_pre_task(tf::Taskflow& flow, PipeCtxT& pipe_ctx,
                         std::function<void(PipeCtxT& pipe_ctx)> pre_fn) {
  return flow.emplace([pre_fn, &pipe_ctx]() { pre_fn(pipe_ctx); });
}

template <typename PipeCtxT>
tf::Task create_post_task(tf::Taskflow& flow, PipeCtxT& pipe_ctx,
                          std::function<void(PipeCtxT& pipe_ctx)> post_fn) {
  return flow.emplace([post_fn, &pipe_ctx]() { post_fn(pipe_ctx); });
}

template <typename PipeCtxT>
tf::Task create_infer_task(tf::Taskflow& flow, PipeCtxT& pipe_ctx) {
  return flow.emplace([&pipe_ctx]() {
    auto input_name = pipe_ctx.infer->getInputName(0);
    pipe_ctx.infer->setInputTensor(input_name, pipe_ctx.input_tensor);
    pipe_ctx.infer->run();
    std::cout << "done!" << std::endl;
    for (int index = 0; index < pipe_ctx.output_tensors.size(); index++) {
      auto output_name = pipe_ctx.infer->getOutputName(index);
      pipe_ctx.output_tensors[index] =
          pipe_ctx.infer->getOutputTensor(output_name);
    }
  });
}

template <typename PipeCtxT>
std::tuple<tf::Task, tf::Task, tf::Task> create_nn_pipe(
    tf::Taskflow& flow, PipeCtxT& pipe_ctx,
    std::function<void(PipeCtxT& pipe_ctx)> pre_fn,
    std::function<void(PipeCtxT& pipe_ctx)> post_fn) {
  auto pre_task = create_pre_task(flow, pipe_ctx, pre_fn);
  auto infer_task = create_infer_task(flow, pipe_ctx);
  auto post_task = create_post_task(flow, pipe_ctx, post_fn);
  pre_task.precede(infer_task);
  infer_task.precede(post_task);
  return std::make_tuple(pre_task, infer_task, post_task);
}
