cmake_minimum_required(VERSION 3.16.3)

project(AIxeleratorService VERSION 1.00 LANGUAGES CXX C Fortran)



set(CMAKE_CXX_STANDARD 17)
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1")
add_compile_options("-g")

# all dependent libraries will be installed to ${CMAKE_INSTALL_PREFIX}/bin, so that they can be found via this RPATH setting
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

include(FetchContent)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Found CUDA compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDAToolkit: ${CUDAToolkit_VERSION}")
        # PyTorch's CMake still relies on legacy FindCUDA variables in parts of
        # its CUDA detection path (notably CUDA_INCLUDE_DIRS for a try_run
        # header-version probe). Mirror the modern toolkit result into those
        # variables so torch's probe sees <cuda.h> without hardcoding paths.
        set(_aix_cuda_root "")
        if(DEFINED CUDAToolkit_ROOT_DIR AND NOT CUDAToolkit_ROOT_DIR STREQUAL "")
            set(_aix_cuda_root "${CUDAToolkit_ROOT_DIR}")
        elseif(DEFINED CMAKE_CUDA_COMPILER_TOOLKIT_ROOT AND NOT CMAKE_CUDA_COMPILER_TOOLKIT_ROOT STREQUAL "")
            set(_aix_cuda_root "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
        endif()

        if(NOT _aix_cuda_root STREQUAL "")
            set(CUDA_TOOLKIT_ROOT_DIR "${_aix_cuda_root}" CACHE PATH "CUDA toolkit root for legacy FindCUDA consumers" FORCE)
            set(CUDA_INCLUDE_DIRS "${_aix_cuda_root}/include" CACHE PATH "CUDA include directory for legacy FindCUDA consumers" FORCE)
        endif()
    endif()

    # determine individual MAJOR, MINOR, PATCH version of CUDA
    string(REPLACE "." ";" CUDA_VERSION_LIST "${CMAKE_CUDA_COMPILER_VERSION}")
    list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
    list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
    list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)
else()
    message(STATUS "CUDA language not available on this system.")
endif()

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE Release)
#add_compile_options("-fsanitize=address")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# Options for AI-Frameworks
option(WITH_TORCH "Build with support for Torch backend" OFF)
set(TORCH_VERSION "2.7.1" CACHE STRING "Version of prebuilt Torch to use. Version format: <MAJOR>.<MINOR>.<PATCH>")
set(LIBTORCH_DIR "" CACHE PATH "Optional path to an existing libtorch CMake prefix (contains share/cmake/Torch)")
option(USE_PYTHON_TORCH_CMAKE_PREFIX "Use torch.utils.cmake_prefix_path from the AIxelerator Python venv before downloading libtorch" ON)
option(WITH_TENSORFLOW "Build with support for Tensorflow backend" OFF)
set(TENSORFLOW_VERSION "2.17.0" CACHE STRING "Version of prebuilt TensorFlow to use.")
option(WITH_SOL "Build with support for NEC SOL backend" OFF)
option(WITH_ONNX "Build with support for ONNX backend" OFF)
set(ONNXRUNTIME_VERSION "1.22.0" CACHE STRING "Version of prebuilt ONNXRuntime to use.")
option(WITH_SCOREP "Build with Score-P instrumentation for profiling/tracing" OFF)
option(WITH_HWLOC "Build with HWLOC and therefore GPU affine RR distribution" OFF)
option(WITH_FORTRAN_INTERFACE "Build Fortran interface to use AIxeleratorService in Fortran programs." OFF)

if(WITH_FORTRAN_INTERFACE)
  include(FortranCInterface)
  FortranCInterface_VERIFY(CXX)
  set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
endif()

if(CMAKE_CUDA_COMPILER)
  if(WITH_TORCH)
    # define architecture for Torch, e.g. "cu126" (CUDA 12.6) or "cpu"
    set(TORCH_ARCH "cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}")
  endif()
  if(WITH_ONNX)
    set(ONNX_ARCH "-gpu")
  endif()
else()
  if(WITH_TORCH)
    set(TORCH_ARCH "cpu") 
  endif()
  if(WITH_ONNX)
    set(ONNX_ARCH "") # onnx does not explicitly name their libraries "cpu" but omits the "gpu" term
  endif()
endif()

# add error check if no AI framework is activated
if((NOT ${WITH_TORCH} AND NOT ${WITH_TENSORFLOW}) AND NOT ${WITH_SOL} AND NOT ${WITH_ONNX})
  message(FATAL_ERROR "You are trying to build the AIxeleratorService without any ML framework backend. At least one of the following options need to be defined: \n -DWITH_TORCH=ON for PyTorch libtorch backend \n -DWITH_TENSORFLOW=ON for TensorFlow backend \n -DWITH_ONNX=ON for ONNX runtime backend")
endif()

# general packages
find_package(MPI REQUIRED)
if(WITH_SCOREP)
  find_package(Scorep REQUIRED)
endif()

if(WITH_HWLOC AND NOT DEFINED HWLOC_ROOT)
    message(ERROR "HWLOC enabled but $HWLOC_ROOT not defined! Please set HWLOC_ROOT to root directory of your HWLOC installation.")
endif()
if(WITH_HWLOC AND NOT DEFINED NUMA_ROOT)
    essage(ERROR "HWLOC enabled but $NUMA_ROOT not defined! Please set NUMA_ROOT to root directory of your NUMACTL installation.")
endif()
if(WITH_HWLOC)
    add_compile_options("-DWITH_HWLOC")
    message(STATUS "HWLOC enabled. Using HWLOC_ROOT=${HWLOC_ROOT} and NUMA_ROOT=${NUMA_ROOT}")
endif()

# creation of virtual environment
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if (NOT AIX_VENV_DIR)
  set(AIX_VENV_DIR "${CMAKE_BINARY_DIR}/extern/venv-py${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}${Python3_VERSION_PATCH}")
endif()
message(STATUS "Using AIX_VENV_DIR: ${AIX_VENV_DIR}")

# check if the environment already exists
if(NOT EXISTS "${AIX_VENV_DIR}/bin/activate" AND NOT AIX_SKIP_VENV_CREATION)
  message(STATUS "No existing Python virtual environment found! Installing now!")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv ${AIX_VENV_DIR}
    RESULT_VARIABLE VENV_RESULT
    COMMAND_ECHO STDOUT
  )
  if(NOT VENV_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to create Python virtual environment! Error code: ${VENV_RESULT}")
  endif()

  # install required packages

  # install numpy
  execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install numpy
      RESULT_VARIABLE NUMPY_RESULT
      COMMAND_ECHO STDOUT
  )
  if(NOT NUMPY_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install numpy! Error code: ${NUMPY_RESULT}")
  endif()

  # install pytorch 
  if(WITH_TORCH OR WITH_ONNX) # (also required to create a test model for ONNXRuntime in Python)
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install torch==${TORCH_VERSION}
      RESULT_VARIABLE TORCH_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TORCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to pip install Torch ${TORCH_VERSION}! Error code: ${TORCH_RESULT}")
    endif()
  endif()

  if(WITH_TENSORFLOW)
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install tensorflow==${TENSORFLOW_VERSION}
      RESULT_VARIABLE TF_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TF_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install TensorFlow ${TENSORFLOW_VERSION}! Error code: ${TF_RESULT}")
    endif()

    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install tf-keras==${TENSORFLOW_VERSION}
      RESULT_VARIABLE TF_KERAS_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT TF_KERAS_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install tf-keras ${TENSORFLOW_VERSION}! Error code: ${TF_KERAS_RESULT}")
    endif()
  endif()

  if(WITH_ONNX)
    # pip install onnx
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install onnx
      RESULT_VARIABLE ONNX_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT ONNX_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install onnx ${ONNX_VERSION}! Error code: ${ONNX_RESULT}")
    endif()

    # pip install onnxruntime
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/pip install onnxruntime${ONNX_ARCH}==${ONNXRUNTIME_VERSION}
      RESULT_VARIABLE ONNX_RESULT
      COMMAND_ECHO STDOUT
    )
    if(NOT ONNX_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to pip install ONNXRuntime ${ONNX_VERSION}! Error code: ${ONNX_RESULT}")
    endif()   
  endif()

else()
  message(STATUS "Found existing Python virtual environment: ${AIX_VENV_DIR}")
endif()
###################### END CREATION OF VIRTUAL ENVIRONMENT ######################

if(WITH_TORCH)
  add_compile_options("-DWITH_TORCH")

  set(TORCH_SOURCE_DIR "${CMAKE_BINARY_DIR}/extern/torch")
  message(STATUS "TORCH_SOURCE_DIR = ${TORCH_SOURCE_DIR}")

  # Try existing Torch CMake package first (either explicitly provided or from python torch wheel)
  set(TORCH_CMAKE_PREFIX_CANDIDATE "")
  if(LIBTORCH_DIR)
    set(TORCH_CMAKE_PREFIX_CANDIDATE "${LIBTORCH_DIR}")
    message(STATUS "Trying existing libtorch from LIBTORCH_DIR=${LIBTORCH_DIR}")
  elseif(USE_PYTHON_TORCH_CMAKE_PREFIX)
    execute_process(
      COMMAND ${AIX_VENV_DIR}/bin/python -c "import torch; print(torch.utils.cmake_prefix_path)"
      RESULT_VARIABLE TORCH_PY_PREFIX_RESULT
      OUTPUT_VARIABLE TORCH_PY_PREFIX_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(TORCH_PY_PREFIX_RESULT EQUAL 0 AND NOT TORCH_PY_PREFIX_OUTPUT STREQUAL "")
      set(TORCH_CMAKE_PREFIX_CANDIDATE "${TORCH_PY_PREFIX_OUTPUT}")
      message(STATUS "Trying existing libtorch from python torch cmake prefix: ${TORCH_CMAKE_PREFIX_CANDIDATE}")
    endif()
  endif()

  if(NOT TORCH_CMAKE_PREFIX_CANDIDATE STREQUAL "")
    set(_aix_saved_cpath "$ENV{CPATH}")
    set(_aix_saved_ld_library_path "$ENV{LD_LIBRARY_PATH}")
    if(CMAKE_CUDA_COMPILER)
      set(_aix_cuda_include_dir "")
      set(_aix_cuda_lib_dir "")
      if(DEFINED CUDA_INCLUDE_DIRS AND NOT CUDA_INCLUDE_DIRS STREQUAL "")
        list(GET CUDA_INCLUDE_DIRS 0 _aix_cuda_include_dir)
      elseif(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT CUDA_TOOLKIT_ROOT_DIR STREQUAL "")
        set(_aix_cuda_include_dir "${CUDA_TOOLKIT_ROOT_DIR}/include")
      endif()
      if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT CUDA_TOOLKIT_ROOT_DIR STREQUAL "")
        set(_aix_cuda_lib_dir "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
      endif()
      if(NOT _aix_cuda_include_dir STREQUAL "")
        if(_aix_saved_cpath STREQUAL "")
          set(ENV{CPATH} "${_aix_cuda_include_dir}")
        else()
          set(ENV{CPATH} "${_aix_cuda_include_dir}:$ENV{CPATH}")
        endif()
      endif()
      if(NOT _aix_cuda_lib_dir STREQUAL "")
        if(_aix_saved_ld_library_path STREQUAL "")
          set(ENV{LD_LIBRARY_PATH} "${_aix_cuda_lib_dir}")
        else()
          set(ENV{LD_LIBRARY_PATH} "${_aix_cuda_lib_dir}:$ENV{LD_LIBRARY_PATH}")
        endif()
      endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_CANDIDATE})
    find_package(Torch QUIET)
    set(ENV{CPATH} "${_aix_saved_cpath}")
    set(ENV{LD_LIBRARY_PATH} "${_aix_saved_ld_library_path}")
  endif()

  if(NOT Torch_FOUND)
    message(STATUS "No existing libtorch found, downloading Torch ${TORCH_VERSION} for ${TORCH_ARCH}...")

    # TODO: download libtorch CPU or GPU
    # Torch changed download URL starting with version 2.8.0 and dropped "cxx11-abi" from the URL!
    if(TORCH_VERSION VERSION_GREATER_EQUAL "2.8.0")
      set(TORCH_DOWNLOAD_URL "https://download.pytorch.org/libtorch/${TORCH_ARCH}/libtorch-shared-with-deps-${TORCH_VERSION}%2B${TORCH_ARCH}.zip")
    else()
      set(TORCH_DOWNLOAD_URL "https://download.pytorch.org/libtorch/${TORCH_ARCH}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_ARCH}.zip")
    endif()
    
    FetchContent_Declare(
      TorchDownload
      URL ${TORCH_DOWNLOAD_URL}
      SOURCE_DIR "${TORCH_SOURCE_DIR}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(TorchDownload)

    list(APPEND CMAKE_PREFIX_PATH ${TORCH_SOURCE_DIR})
    find_package(Torch REQUIRED)
  else()
    message(STATUS "Using existing libtorch from CMake package at: ${Torch_DIR}")
  endif()

  # remove libcuda from TORCH_LIBRARIES
  message(STATUS "Torch Libraries defined by FindTorch = ${TORCH_LIBRARIES}")
  set(TORCH_LIBRARIES_TMP "")
  foreach (_lib ${TORCH_LIBRARIES})
    string(FIND "${_lib}" "libcuda.so" LIB_CUDA_CONTAINED)
    if(LIB_CUDA_CONTAINED EQUAL -1)
      if(TORCH_LIBRARIES_TMP STREQUAL "")
        set(TORCH_LIBRARIES_TMP "${_lib}")
      else()
        set(TORCH_LIBRARIES_TMP "${TORCH_LIBRARIES_TMP};${_lib}")
      endif()
    endif()
  endforeach()
  message(STATUS "modified Torch Libraries = ${TORCH_LIBRARIES_TMP}")
  set(TORCH_LIBRARIES ${TORCH_LIBRARIES_TMP})

  # add torch libraries required for installation to be found via RPATH
  get_filename_component(TORCH_PACKAGE_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
  set(TORCH_LIB_DIR "${TORCH_PACKAGE_ROOT}/lib")
  message(STATUS "TORCH_LIB_DIR = ${TORCH_LIB_DIR}")
  file(GLOB TORCH_CORE_LIBS "${TORCH_LIB_DIR}/libtorch*.so")
  message(STATUS "TORCH_CORE_LIBS = ${TORCH_CORE_LIBS}")
  file(GLOB TORCH_C10_LIBS "${TORCH_LIB_DIR}/libc10*.so")
  message(STATUS "TORCH_C10_LIBS = ${TORCH_C10_LIBS}")
  set(TORCH_INSTALL_LIBS ${TORCH_CORE_LIBS} ${TORCH_C10_LIBS})
  message(STATUS "TORCH_INSTALL_LIBS = ${TORCH_INSTALL_LIBS}")

  file(GLOB TORCH_INSTALL_LIBS "${TORCH_LIB_DIR}/*.so*")
  install(
    FILES ${TORCH_INSTALL_LIBS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )

  if(CMAKE_CUDA_COMPILER)
    if(TARGET CUDA::cupti)
      message(STATUS "Linking AIxeleratorService against CUDA::cupti")
      set(AIX_CUPTI_LIB CUDA::cupti)
    else()
      set(_cupti_hints "")
      if(DEFINED CUDAToolkit_LIBRARY_ROOT)
        list(APPEND _cupti_hints ${CUDAToolkit_LIBRARY_ROOT})
      endif()
      if(DEFINED CUDAToolkit_ROOT)
        list(APPEND _cupti_hints ${CUDAToolkit_ROOT})
        list(APPEND _cupti_hints ${CUDAToolkit_ROOT}/extras/CUPTI/lib64)
        list(APPEND _cupti_hints ${CUDAToolkit_ROOT}/lib64)
      endif()
      if(DEFINED CMAKE_CUDA_COMPILER_TOOLKIT_ROOT)
        list(APPEND _cupti_hints ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT})
        list(APPEND _cupti_hints ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/extras/CUPTI/lib64)
        list(APPEND _cupti_hints ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/lib64)
      endif()

      find_library(AIX_CUPTI_LIB
        NAMES cupti
        HINTS ${_cupti_hints}
        PATH_SUFFIXES extras/CUPTI/lib64 lib64
      )
    endif()

    if(AIX_CUPTI_LIB)
      message(STATUS "Using CUPTI library: ${AIX_CUPTI_LIB}")
    else()
      message(FATAL_ERROR "WITH_TORCH is ON with CUDA available, but the CUPTI library could not be found.")
    endif()
  endif()
endif()

if(WITH_TENSORFLOW)
  add_compile_options("-DWITH_TENSORFLOW")
  # only needed for newer versions of TF (e.g. 2.11 but maybe also lower)
  # simulate a findTensorflow.cmake

  set(Tensorflow_DIR "${CMAKE_BINARY_DIR}/extern/tensorflow")
  message(STATUS "Tensorflow_DIR = ${Tensorflow_DIR}")

  # download TensorFlow
  FetchContent_Declare(
    TensorFlowDownload
    URL "https://storage.googleapis.com/tensorflow/versions/${TENSORFLOW_VERSION}/libtensorflow-gpu-linux-x86_64.tar.gz"
    SOURCE_DIR "${Tensorflow_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(TensorFlowDownload)

  set(TENSORFLOW_PROTOBUF_INCLUDE "${AIX_VENV_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/tensorflow/include/")

  set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include ${TENSORFLOW_PROTOBUF_INCLUDE})
  message(STATUS "TENSORFLOW_INCLUDE_DIRS = ${TENSORFLOW_INCLUDE_DIRS}")
  set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
  message(STATUS "TENSORFLOW_LIB_DIR = ${TENSORFLOW_LIB_DIR}")
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
  message(STATUS "TENSORFLOW_LIBRARIES = ${TENSORFLOW_LIBRARIES}")

  # modern CMake solution using IMPORTED targets
  add_library(TensorFlow::tensorflow SHARED IMPORTED GLOBAL)
  set_target_properties(TensorFlow::tensorflow PROPERTIES
    IMPORTED_LOCATION "${TENSORFLOW_LIB_DIR}/libtensorflow.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_INCLUDE_DIRS}"
    IMPORTED_SONAME "libtensorflow.so"
    INTERFACE_LINK_DIRECTORIES ${TENSORFLOW_LIB_DIR}
  )

  # modern CMake solution using IMPORTED targets
  add_library(TensorFlow::framework SHARED IMPORTED GLOBAL)
  set_target_properties(TensorFlow::framework PROPERTIES
    IMPORTED_LOCATION "${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so"
    IMPORTED_SONAME "libtensorflow_framework.so"
    INTERFACE_LINK_DIRECTORIES "${TENSORFLOW_LIB_DIR}"
  )

  file(GLOB TENSORFLOW_CORE_LIBS "${TENSORFLOW_LIB_DIR}/libtensorflow*.so*")
  file(GLOB TENSORFLOW_FRAMEWORK_LIBS "${TENSORFLOW_LIB_DIR}/libtensorflow_framework*.so*")
  set(TENSORFLOW_SHARED_LIBS ${TENSORFLOW_CORE_LIBS} ${TENSORFLOW_FRAMEWORK_LIBS})

  install(
    FILES
      ${TENSORFLOW_SHARED_LIBS}
    DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib
  )
endif()

if(WITH_ONNX)
  add_compile_options("-DWITH_ONNX")

  # URL 
  set(ONNX_DIR "${CMAKE_BINARY_DIR}/extern/onnxruntime")
  message(STATUS "ONNX_DIR = ${ONNX_DIR}")

  # download TensorFlow
  FetchContent_Declare(
    ONNXRuntimeDownload
    URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64${ONNX_ARCH}-1.22.0.tgz"
    SOURCE_DIR "${ONNX_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(ONNXRuntimeDownload)

  message(STATUS "ONNX_DIR = ${ONNX_DIR}")
  set(ONNX_INCLUDE_DIRS ${ONNX_DIR}/include)
  message(STATUS "ONNX_INCLUDE_DIRS = ${ONNX_INCLUDE_DIRS}")
  set(ONNX_LIB_DIR ${ONNX_DIR}/lib)
  message(STATUS "ONNX_LIB_DIR = ${ONNX_LIB_DIR}")
  set(ONNX_LIBRARIES ${ONNX_LIB_DIR}/libonnxruntime.so)
  message(STATUS "ONNX_LIBRARIES = ${ONNX_LIBRARIES}")

  # modern CMake solution using IMPORTED targets
  add_library(ONNXRuntime::onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNX_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_INCLUDE_DIRS}"
    IMPORTED_SONAME "libonnxruntime.so"
    INTERFACE_LINK_DIRECTORIES ${ONNX_LIB_DIR}
  )

  file(GLOB ONNX_INSTALL_LIBS "${ONNX_LIB_DIR}/libonnxruntime*.so*")
  install(
    FILES
      ${ONNX_INSTALL_LIBS}
    DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib
  )
endif()

if(WITH_SOL)
  add_compile_options("-DWITH_SOL")
  message(STATUS "VEDA_DIR = ${VEDA_DIR}") 
  set(VEDA_INCLUDE_DIRS ${VEDA_DIR}/include)
  message(STATUS "VEDA_INCLUDE_DIRS = ${VEDA_INCLUDE_DIRS}")
  set(VEDA_LIBRARIES ${VEDA_DIR}/lib64/libveda.so.0)
  message(STATUS "VEDA_LIBRARIES = ${VEDA_LIBRARIES}")
endif()

if(WITH_SCOREP)
  add_compile_options("-DSCOREP")
  message(STATUS "AIxeleratorService will be build with Score-P intrumentation!")
endif()

add_subdirectory(src)
add_subdirectory(models)


if(BUILD_TESTS)
  add_subdirectory(test)
endif()
