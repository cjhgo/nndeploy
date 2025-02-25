#pragma once
#include "nndep.h"

namespace pipeline{
using namespace nndeploy;
enum class Platform {
  LinuxAarch64,
  LinuxArm,
  LinuxX86_64,
  AppleX86_64,
  AppleArm64
};

template <Platform P> struct PlatformTraits {
  static constexpr base::ModelType model_type = base::kModelTypeRknn;
  static constexpr base::InferenceType infer_type = base::kInferenceTypeRknn;
  static constexpr const char *suffix = "rknn";
};

template <> struct PlatformTraits<Platform::LinuxX86_64> {
  static constexpr base::ModelType model_type = base::kModelTypeOnnx;
  static constexpr base::InferenceType infer_type =
      base::kInferenceTypeOnnxRuntime;
  static constexpr const char *suffix = "onnx";
};

template <> struct PlatformTraits<Platform::AppleX86_64> {
  static constexpr base::ModelType model_type = base::kModelTypeOnnx;
  static constexpr base::InferenceType infer_type =
      base::kInferenceTypeOnnxRuntime;
  static constexpr const char *suffix = "onnx";
};

template <> struct PlatformTraits<Platform::AppleArm64> {
  static constexpr base::ModelType model_type = base::kModelTypeOnnx;
  static constexpr base::InferenceType infer_type =
      base::kInferenceTypeOnnxRuntime;
  static constexpr const char *suffix = "onnx";
};

template <Platform P> struct InferParamT {
  InferParamT(const std::string &name)
      : name(name), model_stem(name), model_type(PlatformTraits<P>::model_type),
        infer_type(PlatformTraits<P>::infer_type),
        suffix(PlatformTraits<P>::suffix) {

    std::string model_folder = "/tmp/";
    model_path = model_folder + model_stem + "." + suffix;
    // auto cfg = klite::utils::GCfg::getGlobalCfg();
    // auto model_folder = cfg.getKey<std::string>("models.model_folder");
    // auto model_stem =
    //     cfg.getKey<std::string>(fmt::format("models.{}.model_stem", name));
    // model_path =
    //     fmt::format("{}/{}/{}.{}", model_folder, suffix, model_stem, suffix);
    // in_name = cfg.getKey<std::string>(
    //     fmt::format("models.{}.{}.in_name", name, suffix));
    // outs_name = cfg.getArray<std::string>(
    //     fmt::format("models.{}.{}.outs_name", name, suffix));
  }

  std::string name;
  std::string model_stem;
  std::string model_path;
  // std::string in_name;
  // std::vector<std::string> outs_name;
  base::ModelType model_type;
  base::InferenceType infer_type;
  std::string suffix;
};

using InferParam = InferParamT<Platform::CURRENT_PLATFORM>;

inline std::shared_ptr<nndeploy::inference::Inference> create_inference(
    std::string stem,
    std::function<void(nndeploy::base::Param*)> custom_param = nullptr) {
  InferParam infer_param(stem);
  auto infer = std::shared_ptr<nndeploy::inference::Inference>(
      nndeploy::inference::createInference(infer_param.infer_type));
  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = true;
  inference_param->model_value_ = {infer_param.model_path};
  inference_param->model_type_ = infer_param.model_type;

  auto param = infer->getParam();
  if (custom_param != nullptr) {
    custom_param(param);
  }
  infer->init();
  return infer;
}
}// end namespace pipeline