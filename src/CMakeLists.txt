add_library(AIxeleratorServiceLib SHARED)
set_target_properties(AIxeleratorServiceLib PROPERTIES OUTPUT_NAME AIxeleratorService)
target_sources(
    AIxeleratorServiceLib 
        PRIVATE 
            distributionStrategy/roundRobinDistribution.cpp  
            communicationStrategy/collectiveCommunication.cpp
            communicationStrategy/nonBlockingPtoPCommunication.cpp
            aixeleratorService/aixeleratorService.cpp
            aixeleratorService/aixeleratorService_interfaceC.cpp
)
if(WITH_FORTRAN_INTERFACE)
target_sources(
    AIxeleratorServiceLib 
        PRIVATE 
            aixeleratorService/aixeleratorService_interfaceF.f90
)
endif()

target_include_directories(
    AIxeleratorServiceLib
    PUBLIC  ${PROJECT_SOURCE_DIR}/src/distributionStrategy
            ${PROJECT_SOURCE_DIR}/src/communicationStrategy
            ${PROJECT_SOURCE_DIR}/src/inferenceStrategy
            ${PROJECT_SOURCE_DIR}/src/utils
            ${MPI_INCLUDE_PATH}     
            ${PROJECT_SOURCE_DIR}/src/
)

find_package(Threads REQUIRED)

target_link_libraries(
    AIxeleratorServiceLib 
    PUBLIC ${MPI_LIBRARIES}
           Threads::Threads
)

if(WITH_SCOREP)
    scorep_instrument_target(
        AIxeleratorServiceLib
            USER ON
            COMPILER OFF
            MEMORY none
            MPP mpi
            THREADING none
    )
    target_compile_definitions(AIxeleratorServiceLib PRIVATE SCOREP SCOREP_USER_ENABLE)
    target_link_libraries(AIxeleratorServiceLib PUBLIC nvidia-ml)
endif()

if(WITH_HWLOC)
    target_sources(AIxeleratorServiceLib PRIVATE distributionStrategy/gpuAffineRRDistribution.cpp)
    # Resolve the libraries via find_library: distributions differ in whether
    # they use <root>/lib or <root>/lib64 (e.g. RH9 keeps libnuma in /usr/lib64).
    find_library(HWLOC_LIBRARY NAMES hwloc HINTS ${HWLOC_ROOT}/lib)
    find_library(NUMA_LIBRARY NAMES numa HINTS ${NUMA_ROOT}/lib ${NUMA_ROOT}/lib64)
    if(NOT HWLOC_LIBRARY OR NOT NUMA_LIBRARY)
        message(FATAL_ERROR "WITH_HWLOC=ON but libhwloc/libnuma could not be found (HWLOC_ROOT=${HWLOC_ROOT}, NUMA_ROOT=${NUMA_ROOT})")
    endif()
    target_include_directories(AIxeleratorServiceLib PRIVATE ${HWLOC_ROOT}/include ${NUMA_ROOT}/include)
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${HWLOC_LIBRARY} ${NUMA_LIBRARY})
endif()

if(WITH_TORCH)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/torchInference/torchInference.cpp)
    target_include_directories(AIxeleratorServiceLib PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${TORCH_LIBRARIES})
    if(CMAKE_CUDA_COMPILER)
        target_compile_definitions(AIxeleratorServiceLib PRIVATE AIX_HAS_CUDA_RUNTIME)
    endif()
    if(AIX_CUPTI_LIB)
        target_link_libraries(AIxeleratorServiceLib PUBLIC ${AIX_CUPTI_LIB})
    endif()
endif()

if(WITH_TENSORFLOW)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/tensorflowInference/tensorflowInference.cpp)
    #target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/tensorflowInference/tensorflowInferenceCpp.cpp)

    # target_include_directories(AIxeleratorServiceLib PUBLIC ${TENSORFLOW_INCLUDE_DIRS}) # not necessary anymore when using imported targets
    target_link_libraries(AIxeleratorServiceLib PUBLIC 
        TensorFlow::tensorflow 
        TensorFlow::framework)
endif()

if(WITH_ONNX)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/onnxInference/onnxInference.cpp)
    #target_include_directories(AIxeleratorServiceLib PUBLIC ${ONNX_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ONNXRuntime::onnxruntime)
endif()

if(WITH_SOL)
    target_sources(AIxeleratorServiceLib PRIVATE inferenceStrategy/solInference/solInference.cpp)
    target_include_directories(AIxeleratorServiceLib PUBLIC ${VEDA_INCLUDE_DIRS})
    target_link_libraries(AIxeleratorServiceLib PUBLIC ${VEDA_LIBRARIES})
endif()

set_target_properties(
    AIxeleratorServiceLib 
    PROPERTIES 
    #    PUBLIC_HEADER 
    #        "aixeleratorService/aixeleratorService.h;aixeleratorService/#aixeleratorService_interfaceC.h;aixeleratorService/#aixeleratorService_interfaceF.f90"
        LINKER_LANGUAGE Fortran
)

# install headers
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    FILES_MATCHING 
        PATTERN "*.h"
)

# TODO: check if required?
# install(
#     FILES ${PROJECT_SOURCE_DIR}/src/aixeleratorService/aixeleratorService_interfaceF.f90
#     DESTINATION "${CMAKE_BINARY_DIR}/include/aixeleratorService"
# )

# install Fortran module
if(WITH_FORTRAN_INTERFACE)
    install(
        FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/aixelerator_service_mod.mod
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/aixeleratorService
    )
endif()

# install shared library
install(
    TARGETS AIxeleratorServiceLib
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    #PUBLIC_HEADER DESTINATION ${CMAKE_BINARY_DIR}/include
)
