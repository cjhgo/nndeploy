#include "infer_task.h"
#include <opencv2/opencv.hpp>


struct YoloCtx{
  nndeploy::inference::Inference* infer = nullptr;
  cv::Mat input;
  nndeploy::device::Tensor* input_tensor;
  std::vector<nndeploy::device::Tensor*> output_tensors = {nullptr};
  ~YoloCtx(){
    // delete yolo;
    delete input_tensor;
    for(auto& output_tensor : output_tensors){
      delete output_tensor;
    }
  }
};


void yolo_onnx_pre(YoloCtx& ctx) {
  cv::Mat div255;
  cv::resize(ctx.input, div255, cv::Size(640, 640));
  div255.convertTo(div255, CV_32FC1, 1.0 / 255.0);
  cv::cvtColor(div255, div255, cv::COLOR_BGR2RGB);
  nndeploy::device::Device *device = nndeploy::device::getDefaultHostDevice();
  nndeploy::device::TensorDesc desc;
  desc.data_format_ = nndeploy::base::kDataFormatNCHW;
  desc.shape_ = {1, 3, 640, 640};
  ctx.input_tensor = new nndeploy::device::Tensor(device, desc);

  std::vector<cv::Mat> planes(3);
  float* dataptr = (float*)ctx.input_tensor->getData();
  for (size_t i=0; i<3; i++) {
      float* cur_ptr = dataptr + i * 640 * 640;
      planes[i] = cv::Mat(640, 640, CV_32FC1, cur_ptr); // warn: hardcoded type !
  }
  cv::split(div255, planes);
  std::cout << "input tensor size: " << ctx.input_tensor->getShape().size() << std::endl;
}

void custom(nndeploy::base::Param* param){
  auto* infer_param = dynamic_cast<nndeploy::inference::InferenceParam*>(param);
  infer_param->is_path_ = true;
  infer_param->model_value_ = {"/tmp/y11.onnx"};
  infer_param->model_type_ = nndeploy::base::kModelTypeOnnx;
}

void yolo_post(YoloCtx& ctx) {
  std::cout << "output tensor size: " << ctx.output_tensors.size() << std::endl;
  for(auto& output_tensor : ctx.output_tensors){
    auto shape = output_tensor->getShape();
    float* data = (float*)output_tensor->getData();
    std::cout << "shape: ";
    for(auto& s : shape){
      std::cout << s << " ";
    }
    std::cout << std::endl;
    std::cout << "data: ";
    for(int i=0; i<10; i++){
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char** argv) {

  auto img = cv::imread("test.jpg", -1);
  YoloCtx ctx;
  ctx.infer = create_inference(nndeploy::base::kInferenceTypeOnnxRuntime, custom);
  ctx.input = img;


  tf::Executor executor;
  tf::Taskflow taskflow("yolo_pipe");
  create_nn_pipe<YoloCtx>(taskflow, ctx, yolo_onnx_pre, yolo_post);
  executor.run(taskflow).get();
  return 0;
}


