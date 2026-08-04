#include "inferenceStrategy/torchInference/torchInference.h"

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <ATen/ATen.h>
#ifdef AIX_HAS_CUDA_RUNTIME
#include <cuda_runtime_api.h>
#endif

#ifdef SCOREP
#include <scorep/SCOREP_User.h>

SCOREP_USER_REGION_DEFINE( torchInferenceHandle )
SCOREP_USER_REGION_DEFINE( torchForwardHandle )
SCOREP_USER_REGION_DEFINE( h2dCopyHandle )
SCOREP_USER_REGION_DEFINE( d2hCopyHandle )
SCOREP_USER_REGION_DEFINE( cpuChunkSliceHandle )
SCOREP_USER_REGION_DEFINE( cpuChunkAssignHandle )
#endif

template<typename T>
torch::Dtype getTypeFromTemplate()
{
    if (std::is_same<T, float>::value)
    {
        return torch::kFloat32;  
    }

    if (std::is_same<T, double>::value)
    {
        return torch::kFloat64;
    }

    std::cerr << "ERROR: Could not get torch::Dtype from template parameter - defaulting to kFloat16!" << std::endl;
    // TODO(fabian): find a nicer way to handle error. Maybe use std::optional as return type?
    return torch::kFloat16;
}

template torch::Dtype getTypeFromTemplate<float>();
template torch::Dtype getTypeFromTemplate<double>();

template<typename T>
void TorchInference<T>::init(
    int batchsize, 
    int device_id, 
    std::string model_file_name, 
    std::vector<int64_t>& input_shape, T* inputData,  
    std::vector<int64_t>& output_shape, T* outputData
){
    device_id_ = device_id;
    model_file_name_ = model_file_name;

    auto dtype = getTypeFromTemplate<T>();

    try{
        torch_model_ = torch::jit::load(model_file_name_);

        // device IDs from 0 to (n-1) represent n GPUs
        if (device_id_ > -1)
        {
            torch_model_.to(torch::Device(torch::kCUDA, device_id_));
        }
        else 
        {   
            // TODO: datatype needs to be templated
            torch_model_.to(dtype);
        }
    }
    catch (const c10::Error& e) {
        throw;
    }

    batchsize_ = batchsize;
    if (batchsize_ < 1)
    {
        std::cerr << "Error in init TorchInference: batchsize should not be zero or negative!" << std::endl;
    }

    const torch::TensorOptions options(dtype);
    
    at::IntArrayRef input_sizes = input_shape;
    input_ = torch::from_blob((void*) inputData, input_sizes, options);

    at::IntArrayRef output_sizes = output_shape;
    output_ = torch::from_blob((void*) outputData, output_sizes, options);
}

template<typename T>
void TorchInference<T>::inference()
{
#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN(torchInferenceHandle, "torchInference::inference", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif

    torch::NoGradGuard no_grad;
    torch_model_.eval();

    static int call_count = 0;
    const char* dump_dir = std::getenv("DUMP_TENSOR_DIR");

    // Dump the full controller input BEFORE slicing
    if (dump_dir) {
        auto cpu = input_.contiguous().to(torch::kCPU);
        std::ostringstream tag; tag << dump_dir << "/aix_full_input_s" << call_count;
        std::ofstream(tag.str() + ".bin", std::ios::binary)
            .write(static_cast<const char*>(cpu.data_ptr()), cpu.numel() * sizeof(T));
        std::ofstream s(tag.str() + ".shape"); s << cpu.dim(); for (int d=0; d<cpu.dim(); ++d) s << " " << cpu.size(d);
    }

    int batch_dim = input_.size(0);
    int num_batches = batch_dim / batchsize_;
    int size_remaining = batch_dim % batchsize_;
    if (size_remaining > 0)
    {
        num_batches++;
    }

    if (device_id_ > -1)
    {
        const bool diagnostics_enabled = std::getenv("AIX_DIAGNOSTICS") != nullptr;
        last_timing_ = {};
#ifdef AIX_HAS_CUDA_RUNTIME
        cudaEvent_t h2d_start, h2d_end, forward_start, forward_end, d2h_start, d2h_end;
        if (diagnostics_enabled) {
            cudaEventCreate(&h2d_start);
            cudaEventCreate(&h2d_end);
            cudaEventCreate(&forward_start);
            cudaEventCreate(&forward_end);
            cudaEventCreate(&d2h_start);
            cudaEventCreate(&d2h_end);
        }
#endif
        for( int i = 0; i < num_batches; i++)
        {
            input_batch_ = input_.slice(0, batchsize_*i, batchsize_*(i+1));
#ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(h2dCopyHandle, "h2d_copy", SCOREP_USER_REGION_TYPE_COMMON)
            #endif
#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) cudaEventRecord(h2d_start);
#endif
            input_gpu_ = input_batch_.to(torch::Device(torch::kCUDA, device_id_));
#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) cudaEventRecord(h2d_end);
#endif
            #ifdef SCOREP
                SCOREP_USER_REGION_END(h2dCopyHandle)
            #endif
            std::vector<torch::jit::IValue> inputs = {input_gpu_};

            if (dump_dir) {
                std::ostringstream tag; tag << dump_dir << "/aix_input_s" << call_count << "_b" << i;
                auto cpu = input_batch_.contiguous().to(torch::kCPU);
                std::ofstream(tag.str() + ".bin", std::ios::binary)
                    .write(static_cast<const char*>(cpu.data_ptr()), cpu.numel() * sizeof(T));
                std::ofstream s(tag.str() + ".shape"); s << cpu.dim(); for (int d=0; d<cpu.dim(); ++d) s << " " << cpu.size(d);
            }

            #ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(torchForwardHandle, "torchInference::forward", SCOREP_USER_REGION_TYPE_COMMON)
            #endif

#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) cudaEventRecord(forward_start);
#endif
            try { output_gpu_ = torch_model_.forward(inputs).toTensor(); } catch (const std::exception& e) { std::cerr << "INPUT SHAPE: "; for(int k=0; k<input_gpu_.dim(); ++k) std::cerr << input_gpu_.size(k) << " "; std::cerr << "\nException: " << e.what() << "\n"; throw; }
#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) cudaEventRecord(forward_end);
#endif
            
            #ifdef SCOREP
                SCOREP_USER_REGION_END(torchForwardHandle)
            #endif

#ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(d2hCopyHandle, "d2h_copy", SCOREP_USER_REGION_TYPE_COMMON)
            #endif
#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) cudaEventRecord(d2h_start);
#endif
            output_.slice(0, batchsize_*i, batchsize_*(i+1)) = output_gpu_.to(torch::kCPU);
#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) {
                cudaEventRecord(d2h_end);
                cudaEventSynchronize(d2h_end);
                float elapsed_ms = 0.0F;
                cudaEventElapsedTime(&elapsed_ms, h2d_start, h2d_end);
                last_timing_.h2d_gpu_ms += elapsed_ms;
                cudaEventElapsedTime(&elapsed_ms, forward_start, forward_end);
                last_timing_.forward_gpu_ms += elapsed_ms;
                cudaEventElapsedTime(&elapsed_ms, d2h_start, d2h_end);
                last_timing_.d2h_gpu_ms += elapsed_ms;
                ++last_timing_.device_batches;
            }
#endif
            #ifdef SCOREP
                SCOREP_USER_REGION_END(d2hCopyHandle)
            #endif
        }
#ifdef AIX_HAS_CUDA_RUNTIME
        if (diagnostics_enabled) {
            cudaEventDestroy(h2d_start);
            cudaEventDestroy(h2d_end);
            cudaEventDestroy(forward_start);
            cudaEventDestroy(forward_end);
            cudaEventDestroy(d2h_start);
            cudaEventDestroy(d2h_end);
        }
#endif
    }
    else
    {
        for( int i = 0; i < num_batches; i++)
        {
#ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(cpuChunkSliceHandle, "cpu_chunk_slice", SCOREP_USER_REGION_TYPE_COMMON)
            #endif
            input_batch_ = input_.slice(0, batchsize_*i, batchsize_*(i+1));
            #ifdef SCOREP
                SCOREP_USER_REGION_END(cpuChunkSliceHandle)
            #endif
            std::vector<torch::jit::IValue> inputs = {input_batch_};

            if (dump_dir) {
                std::ostringstream tag; tag << dump_dir << "/aix_input_s" << call_count << "_b" << i;
                auto cpu = input_batch_.contiguous().to(torch::kCPU);
                std::ofstream(tag.str() + ".bin", std::ios::binary)
                    .write(static_cast<const char*>(cpu.data_ptr()), cpu.numel() * sizeof(T));
                std::ofstream s(tag.str() + ".shape"); s << cpu.dim(); for (int d=0; d<cpu.dim(); ++d) s << " " << cpu.size(d);
            }

            #ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(torchForwardHandle, "torchInference::forward", SCOREP_USER_REGION_TYPE_COMMON)
            #endif

            try { output_batch_ = torch_model_.forward(inputs).toTensor(); } catch (const std::exception& e) { std::cerr << "INPUT SHAPE: "; for(int k=0; k<input_batch_.dim(); ++k) std::cerr << input_batch_.size(k) << " "; std::cerr << "\nException: " << e.what() << std::endl; throw; }

            #ifdef SCOREP
                SCOREP_USER_REGION_END(torchForwardHandle)
            #endif

#ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(cpuChunkAssignHandle, "cpu_chunk_assign", SCOREP_USER_REGION_TYPE_COMMON)
            #endif
            output_.slice(0, batchsize_*i, batchsize_*(i+1)) = output_batch_.to(torch::kCPU);
            #ifdef SCOREP
                SCOREP_USER_REGION_END(cpuChunkAssignHandle)
            #endif
        }   
    }
    call_count++;

#ifdef SCOREP
    SCOREP_USER_REGION_END(torchInferenceHandle)
#endif
}


template class TorchInference<float>;
template class TorchInference<double>;
