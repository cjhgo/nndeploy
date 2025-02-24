
#include "nndeploy/inference/rknn/rknn_inference.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/inference/rknn/rknn_convert.h"
#include "nndeploy/inference/rknn/rknn_inference_param.h"

namespace nndeploy {
namespace inference {

void dump_tensor_attr(rknn_tensor_attr *attr) {
  char dims_str[100];
  memset(dims_str, 0, sizeof(dims_str));
  for (int i = 0; i < attr->n_dims; i++) {
    sprintf(dims_str, "%s%d,", dims_str, attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, "
         "w_stride=%d, size_with_stride=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, dims_str, attr->n_elems,
         attr->size, attr->w_stride, attr->size_with_stride,
         get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

TypeInferenceRegister<TypeInferenceCreator<RknnInference>>
    g_rknn_inference_register(base::kInferenceTypeRknn);

RknnInference::RknnInference(base::InferenceType type) : Inference(type) {}

RknnInference::~RknnInference() {}

base::Status RknnInference::init() {

  base::Status status = base::kStatusCodeOk;
  std::string model_buffer;
  RknnInferenceParam *rknn_inference_param =
      dynamic_cast<RknnInferenceParam *>(inference_param_);
  std::cout << "will load from file" << rknn_inference_param->model_value_[0] << std::endl;
  if (rknn_inference_param->is_path_) {
    model_buffer = base::openFile(rknn_inference_param->model_value_[0]);
  } else {
    model_buffer = rknn_inference_param->model_value_[0];
  }

  if (model_buffer.empty()) {
    NNDEPLOY_LOGET("Load model failed, model buffer empty\n", "RKNN");
    return base::kStatusCodeErrorInferenceRknn;
  }

#ifdef RKNN_TOOLKIT_1
  if (!CHECK_RKNN(rknn_init(
          &rknn_ctx_,
          const_cast<void *>(static_cast<const void *>(model_buffer.data())),
          model_buffer.size(), 0))) {
    return base::kStatusCodeErrorInferenceRknn;
  }
#elif RKNN_TOOLKIT_2
  if (!CHECK_RKNN(rknn_init(
          &rknn_ctx_,
          const_cast<void *>(static_cast<const void *>(model_buffer.data())),
          model_buffer.size(), 0, nullptr))) {
    return base::kStatusCodeErrorInferenceRknn;
  }
#endif
  rknn_sdk_version version;
  auto ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  std::cout << "RKNN SDK version: " << version.api_version << std::endl;
  std::cout << "RKNN SDK version: " << version.drv_version << std::endl;
  rknn_input_output_num io_num;
  if (!CHECK_RKNN(rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num,
                             sizeof(io_num)))) {
    return base::kStatusCodeErrorInferenceRknn;
  };

  rknn_inputs_.resize(io_num.n_input);
  rknn_outputs_.resize(io_num.n_output);
  device::Device *device = device::getDevice(inference_param_->device_type_);
  for (int i = 0; i < io_num.n_input; ++i) {
    rknn_input &input = rknn_inputs_[i];
    rknn_tensor_attr input_attr;
    input_attr.index = i;
    if (!CHECK_RKNN(rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr,
                               sizeof(input_attr)))) {
      return base::kStatusCodeErrorInferenceRknn;
    };
    dump_tensor_attr(&input_attr);
    auto name = std::string(input_attr.name);
    std::cout << name << std::endl;
    base::IntVector shape = RknnConvert::convertToShape(
        input_attr, rknn_inference_param->input_data_format_);
    base::DataType data_type =
        RknnConvert::convertToDataType(rknn_inference_param->input_data_type_);
    base::DataFormat data_format = RknnConvert::convertToDataFormat(
        rknn_inference_param->input_data_format_);

    device::TensorDesc desc;
    desc.shape_ = shape;
    desc.data_type_ = data_type;
    desc.data_format_ = data_format;
    desc.print();

    // create empty buffer tensor for input, use setInputTensor to set buffer
    // later. avoid unnecessray memory copy
    device::Tensor *input_tensor = new device::Tensor(desc, name);
    input_tensors_.insert({name, input_tensor});

    inputs_index_name_[i] = name;

    input.index = i;
    input.type = rknn_inference_param->input_data_type_;
    input.size = input_attr.n_elems * data_type.size();
    std::cout << "input.size: " << input.size << std::endl;
    std::cout << "data type size: " << data_type.size() << std::endl;
    input.fmt = rknn_inference_param->input_data_format_;
    input.pass_through = rknn_inference_param->input_pass_through_;
  }
  for (int i = 0; i < io_num.n_output; ++i) {
    rknn_output &output = rknn_outputs_[i];
    rknn_tensor_attr output_attr;
    output_attr.index = i;
    if (!CHECK_RKNN(rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr,
                               sizeof(output_attr)))) {
      return base::kStatusCodeErrorInferenceRknn;
    };
    dump_tensor_attr(&output_attr);
    auto name = std::string(output_attr.name);
    std::cout << name << std::endl;
    base::IntVector shape = RknnConvert::convertToShape(output_attr);

    device::TensorDesc desc;
    desc.shape_ = shape;
    desc.data_type_ = RknnConvert::convertToDataType(rknn_inference_param->output_data_types_[i]);
    desc.data_format_ = RknnConvert::convertToDataFormat(output_attr.fmt);

    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    output_tensors_.insert({name, output_tensor});

    output.index = i;
    output.want_float = (desc.data_type_ ==  base::dataTypeOf<float>());
    output.is_prealloc = true;
    output.size = output_attr.n_elems * desc.data_type_.size();
    output.buf = output_tensor->getData();
  }
  return status;
}

base::Status RknnInference::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : input_tensors_) {
    delete iter.second;
  }
  input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter.second;
  }
  output_tensors_.clear();
  return status;
}

base::Status RknnInference::run() {
  base::Status status = base::kStatusCodeOk;
  for (int i = 0; i < rknn_inputs_.size(); i++) {
    rknn_inputs_[i].buf = input_tensors_[inputs_index_name_[i]]->getData();
  }
  if (!CHECK_RKNN(rknn_inputs_set(rknn_ctx_, rknn_inputs_.size(),
                                  rknn_inputs_.data()))) {
    return base::kStatusCodeErrorInferenceRknn;
  };
  if (!CHECK_RKNN(rknn_run(rknn_ctx_, nullptr))) {
    return base::kStatusCodeErrorInferenceRknn;
  };
  if (!CHECK_RKNN(rknn_outputs_get(rknn_ctx_, rknn_outputs_.size(),
                                   rknn_outputs_.data(), nullptr))) {
    return base::kStatusCodeErrorInferenceRknn;
  };
  return status;
}

base::Status RknnInference::setInputTensor(const std::string &name,
                                           device::Tensor *input_tensor) {
  base::Status status = base::kStatusCodeOk;

  if (input_tensors_.count(name) > 0) {
    NNDEPLOY_ASSERT(input_tensor->getDesc() == input_tensors_[name]->getDesc());
input_tensor->getDesc().print();
input_tensors_[name]->getDesc().print();
    input_tensors_[name]->justModify(input_tensor->getBuffer());
  } else {
    NNDEPLOY_LOGI("input_tensor nama: %s not exist!\n", name.c_str());
  }
  return status;
}

base::Status RknnInference::reshape(base::ShapeMap &shape_map) {
  return base::kStatusCodeOk;
}

device::Tensor *RknnInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  device::Tensor *internal_tensor = output_tensors_[name];
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  if (flag) {
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    internal_tensor->getBuffer()->copyTo(output_tensor->getBuffer());
    return output_tensor;
  } else {
    device::Tensor *output_tensor =
        new device::Tensor(desc, internal_tensor->getBuffer(), name);
    return output_tensor;
  }
}

}  // namespace inference
}  // namespace nndeploy