add_executable(run ${ROOT_PATH}/piperun/main.cpp)
target_link_libraries(run PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})

add_executable(yolo ${ROOT_PATH}/piperun/yolo_task.cpp)
target_link_libraries(yolo PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})