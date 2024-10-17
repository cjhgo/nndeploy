#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/infer/infer.h"
#include "preprocess.h"


using namespace nndeploy;
int main(int argc, char *argv[])
{

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图graph名称，例如:
  // NNDEPLOY_YOLOV5/NNDEPLOY_YOLOV6/NNDEPLOY_YOLOV8
  std::string name = demo::getName();
  // 推理后端类型，例如:
  // kInferenceTypeOpenVino / kInferenceTypeTensorRt / kInferenceTypeOnnxRuntime
  base::InferenceType inference_type = demo::getInferenceType();
  // 推理设备类型，例如:
  // kDeviceTypeCodeX86:0/kDeviceTypeCodeCuda:0/...
  base::DeviceType device_type = demo::getDeviceType();
  // 模型类型，例如:
  // kModelTypeOnnx/kModelTypeMnn/...
  base::ModelType model_type = demo::getModelType();
  // 模型是否是路径
  bool is_path = demo::isPath();
  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();
  // input path
  std::string input_path = demo::getInputPath();
  // input path
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline / base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();


  // 有向无环图graph的输入边packert
  dag::Edge input("in");
  // 有向无环图graph的输出边packert
  dag::Edge output("out");

  auto* graph = new dag::Graph(name, &input, &output);

  dag::Edge *infer_input = graph->createEdge("input_156");
  dag::Edge *infer_output = graph->createEdge("Sigmoid_/sigmod/Sigmoid/out0_0");
  // dag::Edge *infer_input = graph->createEdge("input");
  // dag::Edge *infer_output = graph->createEdge("output");

  // 解码节点
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", &input);
  decode_node->setPath(input_path);

  graph->addNode(decode_node);


  auto prenode = graph->createNode<models::clfg::PreprocessNode>("preprocess_node", &input, infer_input);
  auto infer = graph->createInfer<infer::Infer>("infer", inference_type, infer_input, infer_output);


  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  auto status = graph->init();
  graph->dump();
  graph->run();

  auto input_img = input.getCvMat(decode_node);
  if(input_img != nullptr) {
    std::cout << input_img->cols << " " << input_img->rows  << input_img->type() << std::endl;
  }
  else{
    std::cout << "input_img is nullptr" << std::endl;
  }

  auto tensor = infer_input->getTensor(prenode);
  std::cout << tensor->getChannel() << " " << tensor->getHeight() << " " << tensor->getWidth() << " " << tensor->getDataType().code_ << " " << tensor->getDataType().bits_ << " " << tensor->getDataType().lanes_ << std::endl;

  tensor = infer_output->getTensor(infer);
  std::cout << tensor->getChannel() << " " << tensor->getHeight() << " " << tensor->getWidth() << " " << tensor->getDataType().code_ << " " << tensor->getDataType().bits_ << " " << tensor->getDataType().lanes_ << std::endl;

  cv::Mat out(160, 160, CV_32FC1, tensor->getData());
  cv::imwrite("out.jpeg", out*255);
  // cv::imwrite("out.jp2", out*255);
  cv::imwrite("out.tiff", out*255);


  graph->deinit();
  delete graph;
  delete decode_node;
  return 0;
}

