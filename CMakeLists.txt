cmake_minimum_required(VERSION 3.23)
project(LibTorchTemplate)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "C:/libtorch")
set(CMAKE_BUILD_TYPE Release)

find_package(Torch REQUIRED)

add_subdirectory("./vendor/tensorboard_logger")

file(GLOB SRC "src/*.c" "src/*.cpp" "src/*.cc" "src/*.h" "src/*.hpp" )
file(GLOB Protobuf_LIBRARIES "${TensorBoard}/Protobuf/cmake/Release/*.lib")
#set(Protobuf_INCLUDE_DIRS "${TensorBoard}/Protobuf/src")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} /MT")

add_executable(LibTorchTemplate ${SRC})

target_link_libraries(LibTorchTemplate "${TORCH_LIBRARIES}" tensorboard_logger)
  #"${TensorBoard}/build/Release/tensorboard_logger.lib")

#target_include_directories(LibTorchTemplate PUBLIC "${TensorBoard}/include")

set_property(TARGET LibTorchTemplate PROPERTY CXX_STANDARD 17)

if (MSVC)
  set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "LibTorchTemplate")
  target_compile_options(LibTorchTemplate PRIVATE /W0)
  target_compile_options(LibTorchTemplate PRIVATE /MD)

  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET LibTorchTemplate
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:LibTorchTemplate>)
endif (MSVC)

add_custom_command(TARGET LibTorchTemplate
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_directory 
                     data
                     $<TARGET_FILE_DIR:LibTorchTemplate>/data)