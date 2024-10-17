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
set(RKSDK_HOME /root/workspace/rv1126_rv1109_sdk/buildroot/output/rockchip_rv1126_rv1109_moqi_m1/)
set(COMMON_PATH ${RKSDK_HOME}/host/arm-buildroot-linux-gnueabihf/sysroot/usr/include CACHE INTERNAL "")
set(USR_PATH ${RKSDK_HOME}/host/arm-buildroot-linux-gnueabihf/sysroot/usr CACHE INTERNAL "")
set(CMAKE_TOOLCHAIN_FILE "${RKSDK_HOME}/host/share/buildroot/toolchainfile.cmake" CACHE STRING "Toolchain file")

message(STATUS "COMMON_PATH: ${COMMON_PATH}")

message(STATUS "USR_PATH: ${USR_PATH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /tmp/m1d CACHE PATH "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY /tmp/m1d CACHE PATH "")

set(CMAKE_INSTALL_PREFIX "${USR_PATH}" CACHE PATH "")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")
set(CMAKE_COLOR_MAKEFILE ON CACHE BOOL "")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "")
# set(CMAKE_BUILD_TYPE Release CACHE STRING "")
# set(CMAKE_COLOR_MAKEFILE OFF CACHE BOOL "")
# set(BUILD_DOC OFF CACHE BOOL "")
# set(BUILD_DOCS OFF CACHE BOOL "")
# set(BUILD_EXAMPLE OFF CACHE BOOL "")
# set(BUILD_EXAMPLES ON CACHE BOOL "")
# set(BUILD_TEST OFF CACHE BOOL "")
# set(BUILD_TESTS OFF CACHE BOOL "")
# set(BUILD_TESTING OFF CACHE BOOL "")
# set(BUILD_SHARED_LIBS ON CACHE BOOL "")

message(STATUS "cmake system name is ${CMAKE_SYSTEM_NAME}")#is empty when load options.cmake
message(STATUS "cmake system processor is ${CMAKE_SYSTEM_PROCESSOR}")#is empty when load options.cmake

# common
set(ENABLE_NNDEPLOY_BUILD_SHARED ON CACHE BOOL "") # 是否编译为动态库，默认ON
set(ENABLE_NNDEPLOY_SYMBOL_HIDE ON CACHE BOOL "") # 符号表是否隐藏，默认为ON
set(ENABLE_NNDEPLOY_CXX17_ABI ON CACHE BOOL "") # C++的版本，选择为C++17，默认为OFF
set(ENABLE_NNDEPLOY_OPENMP ON CACHE BOOL "") # 否使用OpenMP，该选项在Mac/iOS平台无效，默认为ON
set(ENABLE_NNDEPLOY_TIME_PROFILER ON CACHE BOOL "") # 时间性能Profile，默认为ON
set(ENABLE_NNDEPLOY_OPENCV ON CACHE BOOL "") # 是否链接第三方库opencv，默认为OFF
set(NNDEPLOY_OPENCV_LIBS) # 链接的具体的opencv库名称，例如opencv_world480，opencv_java4等

# # base
set(ENABLE_NNDEPLOY_BASE ON CACHE BOOL "") # 是否编译base目录中文件，默认为ON

# # thread
set(ENABLE_NNDEPLOY_THREAD_POOL ON CACHE BOOL "") # 是否编译thread_pool目录中文件，默认为ON


# # device
set(ENABLE_NNDEPLOY_DEVICE ON CACHE BOOL "") # 是否编译device目录中文件，默认为ON
set(ENABLE_NNDEPLOY_DEVICE_CPU ON CACHE BOOL "") # 是否使能device cpu，默认为ON
set(ENABLE_NNDEPLOY_DEVICE_ARM OFF) # 是否使能device arm，默认为OFF

# # ir
set(ENABLE_NNDEPLOY_IR ON CACHE BOOL "") # 是否编译ir目录中文件，默认为OFF
set(ENABLE_NNDEPLOY_IR_ONNX OFF) # 是否编译ir目录中文件，默认为OFF

# # op
set(ENABLE_NNDEPLOY_OP ON CACHE BOOL "") # 是否编译op目录中文件，默认为OFF

# # net
set(ENABLE_NNDEPLOY_NET ON CACHE BOOL "") # 是否编译net目录中文件，默认为OFF

# # inference
set(ENABLE_NNDEPLOY_INFERENCE ON CACHE BOOL "") # 是否编译inference目录中文件，默认为ON
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 ${USR_PATH} CACHE PATH "") # 是否使能INFERENCE RKNN_TOOLKIT_1，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 OFF) # 是否使能INFERENCE RKNN_TOOLKIT_2，默认为OFF

# # dag
set(ENABLE_NNDEPLOY_DAG ON CACHE BOOL "") # 是否编译dag目录中文件，默认为ON

# plugin
set(ENABLE_NNDEPLOY_PLUGIN ON CACHE BOOL "") # 是否编译plugin目录中文件，默认为ON


# demo
set(ENABLE_NNDEPLOY_DEMO ON CACHE BOOL "") # 是否使能可执行程序demo，默认为OFF


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


message(STATUS "demo config is ${ENABLE_NNDEPLOY_DEMO}")
message(STATUS "rknn config is ${ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1}")