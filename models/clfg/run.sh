set -x
abspath="$(realpath $0)"
basepath=${abspath%/*}

$basepath/../../build/clfg --name clfg --inference_type kInferenceTypeOnnxRuntime \
--device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path \
--model_value  \
/Users/junhangc/sideproject/stream_pipe/python/models/2403_cl_fg/assets/k2_fg_exp6_6-300.onnx \
--codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential \
--input_path \
/Users/junhangc/sideproject/stream_pipe/python/models/2403_cl_fg/assets/11_unroll.png \
--output_path /tmp/sample_output.jpg

./clfg --name clfg --inference_type kInferenceTypeRknn \
--device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeRknn --is_path \
--model_value  /tmp/lib/out.rknn \
--codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential \
--input_path  /tmp/lib/in.png \
--output_path /tmp/sample_output.png
