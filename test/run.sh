set -x
abspath="$(realpath $0)"
basepath=${abspath%/*}
$basepath/../build/nndeploy_demo_detect --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOnnxRuntime \
--device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path \
--model_value /Users/junhangc/sideproject/nndeploy/test/yolov8n.onnx \
--codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential \
--input_path /Users/junhangc/sideproject/stream_pipe/python/models/m2409_yolov8/data/bus.jpg \
--output_path /tmp/sample_output.jpg

# --input_path /Users/junhangc/sideproject/nndeploy/test/test_data_detect_sample.jpg \