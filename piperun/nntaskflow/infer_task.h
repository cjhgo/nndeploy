#pragma once
#include "nndep.h"
#include "taskflow/taskflow.hpp"  // Taskflow is header-only

// struct YoloCtx{
//   std::shared_ptr<nndeploy::inference::Inference> infer;
//   cv::Mat input;
//   nndeploy::device::Tensor input_tensor;
//   std::vector<std::unique_ptr<nndeploy::device::Tensor>> output_tensors;
//   std::vector<pipeline::yolo_utils::YoloBox> boxes;
// };
namespace pipeline {


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
    pipe_ctx.infer->setInputTensor(input_name, &pipe_ctx.input_tensor);
    pipe_ctx.infer->run();
    pipe_ctx.output_tensors.resize(pipe_ctx.infer->getNumOfOutputTensor());
    std::cout << "done!" << std::endl;
    for (int index = 0; index < pipe_ctx.output_tensors.size(); index++) {
      auto output_name = pipe_ctx.infer->getOutputName(index);
      pipe_ctx.output_tensors[index].reset(
          pipe_ctx.infer->getOutputTensorAfterRun(
              output_name, nndeploy::device::getDefaultHostDeviceType(), true));
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

}  // namespace pipeline