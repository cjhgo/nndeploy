# add_executable(run ${ROOT_PATH}/piperun/main.cpp)
# target_link_libraries(run PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})
include_directories(${ROOT_PATH}/piperun)
file(GLOB_RECURSE SOURCE
  "${ROOT_PATH}/piperun/yolo-utils/*.cc"
)
add_executable(yolo ${ROOT_PATH}/piperun/yolov8-segp/main.cpp ${SOURCE})
target_link_libraries(yolo PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})

add_executable(plate ${ROOT_PATH}/piperun/yolov8-plate/main.cpp ${SOURCE})
target_link_libraries(plate PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})