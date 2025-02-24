#include <opencv2/opencv.hpp>
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/common.h"
#include "nndeploy/device/device.h"
#include "nndeploy/base/type.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/rknn/rknn_inference_param.h"
#include "taskflow/taskflow.hpp"  // Taskflow is header-only
#include <iostream>

using namespace nndeploy;
int main(){

  tf::Executor executor;
  tf::Taskflow taskflow;

  auto [A, B, C, D] = taskflow.emplace(  // create four tasks
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; }
  );

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  executor.run(taskflow).wait();



  auto rknn_infer = inference::createInference(base::kInferenceTypeRknn);
  auto param = dynamic_cast<inference::RknnInferenceParam *>(rknn_infer->getParam());
  param->is_path_ = true;
  param->model_value_ = {"/root/segp_cut.rknn"};
  param->model_type_ = base::kModelTypeRknn;
  std::cout << std::endl;
  std::cout << param->input_data_type_ << std::endl;
  std::cout << param->output_data_types_.size() << std::endl;
  std::cout << param->scales_.size() << std::endl;
  rknn_infer->init();

  return 0;
}