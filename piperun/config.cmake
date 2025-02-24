add_executable(run ${ROOT_PATH}/piperun/main.cpp)
target_link_libraries(run PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})