cmake_minimum_required(VERSION 3.23)
project(LibTorchTemplate)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "C:/libtorch")
set(CMAKE_BUILD_TYPE Release)

find_package(Torch REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(LibTorchTemplate src/main.cpp)

target_link_libraries(LibTorchTemplate "${TORCH_LIBRARIES}")

set_property(TARGET LibTorchTemplate PROPERTY CXX_STANDARD 14)

if (MSVC)
  set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "LibTorchTemplate")
  target_compile_options(LibTorchTemplate PRIVATE /W0)

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