message(STATUS "Building nndeploy models")

nndeploy_option(ENABLE_NNDEPLOY_MODEL_CLFG "ENABLE_NNDEPLOY_MODEL_CLFG" OFF)
# framework
if(ENABLE_NNDEPLOY_MODEL_CLFG)
  include(${ROOT_PATH}/models/clfg/config.cmake)
endif()
  include(${ROOT_PATH}/models/oldenh/config.cmake)