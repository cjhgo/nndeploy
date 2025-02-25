#include "piperun.h"
#include <opencv2/opencv.hpp>

using namespace pipeline;



void yolo_post(YoloCtx& ctx) {
  pipeline::yolo_utils::non_max_suppression(&ctx.boxes, ctx.output_tensors[0].get());
  for(auto& box : ctx.boxes){
    cv::rectangle(ctx.input, box.rect, cv::Scalar(0, 255, 0), 2);
  }
  cv::imwrite("output.jpg", ctx.input);
}

int main(int argc, char** argv) {

  auto img = cv::imread("test.jpg", -1);
  YoloCtx ctx;
  ctx.infer = pipeline::create_inference("y11");
  ctx.input = img;


  tf::Executor executor;
  tf::Taskflow taskflow("yolo_pipe");
  create_nn_pipe<YoloCtx>(taskflow, ctx, yolo_utils::yolo_pre_process<YoloCtx>, yolo_post);
  executor.run(taskflow).get();
  return 0;
}


