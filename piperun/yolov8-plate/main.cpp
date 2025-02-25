#include <opencv2/opencv.hpp>

#include "piperun.h"

using namespace pipeline;


void yolo_post(YoloCtx& ctx) {
  ctx.output_tensors[0]->getDesc().print();
  pipeline::yolo_utils::non_max_suppression(&ctx.boxes,
                                            ctx.output_tensors[0].get(), 2);

  cv::Mat save;
  cv::resize(ctx.input, save, cv::Size(640, 640));
  for (auto& box : ctx.boxes) {
    cv::rectangle(save, box.rect, cv::Scalar(0, 255, 0), 2);
  }
  cv::imwrite("output.jpg", save);
}

int main(int argc, char** argv) {
  auto img = cv::imread("/tmp/plate.jpg", -1);
  YoloCtx ctx;
  ctx.infer = pipeline::create_inference("plate");
  ctx.input = img;

  tf::Executor executor;
  tf::Taskflow taskflow("yolo_plate");
  create_nn_pipe<YoloCtx>(
      taskflow, ctx,
      yolo_utils::yolo_pre_process<YoloCtx>, yolo_post);
  executor.run(taskflow).get();
  return 0;
}
