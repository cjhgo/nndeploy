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

  std::string name = "oenh";
  base::InferenceType inference_type = base::kInferenceTypeRknn;
  base::DeviceType device_type;
  base::ModelType model_type = base::kModelTypeRknn;
  bool is_path = true;
  std::vector<std::string> model_value = {"./v0.rknn"};
  std::string input_path = "./in.png";
  base::CodecFlag codec_flag = base::kCodecFlagImage;
  base::ParallelType pt = base::kParallelTypeSequential;


  // 有向无环图graph的输入边packert
  dag::Edge input("in");
  // 有向无环图graph的输出边packert
  dag::Edge output("out");

  auto* graph = new dag::Graph(name, &input, &output);

  // dag::Edge *infer_input = graph->createEdge("cjh");
  // dag::Edge *infer_output = graph->createEdge("GPU_1/sl_net/output/Relu:0");

  dag::Edge *infer_input = graph->createEdge("x/out0_1");
  dag::Edge *infer_output = graph->createEdge("GPU_1/sl_net/output/Relu/out0_0");
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
  std::cout << "output: " << tensor->getChannel() << " " << tensor->getHeight() << " " << tensor->getWidth() << " " << tensor->getDataType().code_ - 0<< " " << tensor->getDataType().bits_ - 0 << " " << tensor->getDataType().lanes_ - 0 << std::endl;

  cv::Mat out(640, 640, CV_32FC1, tensor->getData());
  cv::imwrite("out.jpg", out*255);


  graph->deinit();
  delete graph;
  delete decode_node;
  return 0;
}

