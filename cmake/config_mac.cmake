# --------------------------------------------------------------------
# Template custom cmake config for compiling
#
# This file is used to override the build sets in build.
# If you want to change the config, please use the following
# steps. Assume you are off the root directory. First copy the this
# file so that any local changes will be ignored by git
#
# $ mkdir build
# $ cp cmake/config.cmake build
# $ cd build
# $ cmake ..
# $ make -j8
# --------------------------------------------------------------------
# common
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(ENABLE_NNDEPLOY_BUILD_SHARED ON CACHE BOOL "") # 是否编译为动态库，默认ON
set(ENABLE_NNDEPLOY_SYMBOL_HIDE ON CACHE BOOL "") # 符号表是否隐藏，默认为ON
set(ENABLE_NNDEPLOY_CXX17_ABI ON CACHE BOOL "") # C++的版本，选择为C++17，默认为OFF
set(ENABLE_NNDEPLOY_TIME_PROFILER ON CACHE BOOL "") # 时间性能Profile，默认为ON
set(ENABLE_NNDEPLOY_OPENCV ON CACHE BOOL "") # 是否链接第三方库opencv，默认为OFF
set(NNDEPLOY_OPENCV_LIBS) # 链接的具体的opencv库名称，例如opencv_world480，opencv_java4等

# # base
set(ENABLE_NNDEPLOY_BASE ON CACHE BOOL "") # 是否编译base目录中文件，默认为ON

# # thread
set(ENABLE_NNDEPLOY_THREAD_POOL ON CACHE BOOL "") # 是否编译thread_pool目录中文件，默认为ON

# # cryption
set(ENABLE_NNDEPLOY_CRYPTION OFF) # 是否编译crytion目录中文件，默认为ON

# # device
set(ENABLE_NNDEPLOY_DEVICE ON CACHE BOOL "") # 是否编译device目录中文件，默认为ON
set(ENABLE_NNDEPLOY_DEVICE_CPU ON CACHE BOOL "") # 是否使能device cpu，默认为ON

# # ir
set(ENABLE_NNDEPLOY_IR ON CACHE BOOL "") # 是否编译ir目录中文件，默认为OFF

# # op
set(ENABLE_NNDEPLOY_OP ON CACHE BOOL "") # 是否编译op目录中文件，默认为OFF

# # net
set(ENABLE_NNDEPLOY_NET ON CACHE BOOL "") # 是否编译net目录中文件，默认为OFF

# # inference
set(ENABLE_NNDEPLOY_INFERENCE ON CACHE BOOL "") # 是否编译inference目录中文件，默认为ON
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME /opt/homebrew CACHE PATH "") # 是否使能INFERENCE ONNXRUNTIME，默认为OFF

# # dag
set(ENABLE_NNDEPLOY_DAG ON CACHE BOOL "") # 是否编译dag目录中文件，默认为ON

# plugin
set(ENABLE_NNDEPLOY_PLUGIN ON CACHE BOOL "") # 是否编译plugin目录中文件，默认为ON

# test
set(ENABLE_NNDEPLOY_TEST OFF) # 是否使能单元测试，默认为OFF

# demo
set(ENABLE_NNDEPLOY_DEMO ON CACHE BOOL "") # 是否使能可执行程序demo，默认为OFF

# enable python api
set(ENABLE_NNDEPLOY_PYTHON OFF) # ON 表示构建nndeploy的python接口

# plugin
# # preprocess
set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON CACHE BOOL "") # 是否编译plugin目录中文件，默认为ON

# # infer
set(ENABLE_NNDEPLOY_PLUGIN_INFER ON CACHE BOOL "") # 是否编译plugin目录中文件，默认为ON

# # codec
set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON CACHE BOOL "") # 是否编译plugin目录中文件，默认为ON

# # detect
set(ENABLE_NNDEPLOY_PLUGIN_DETECT ON CACHE BOOL "")
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR ON CACHE BOOL "")
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO ON CACHE BOOL "")