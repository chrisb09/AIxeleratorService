#include "inferenceStrategy/torchInference/torchInference.h"

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <array>

#include <ATen/ATen.h>
#include "utils/p2pTimeline.h"
#ifdef AIX_HAS_CUDA_RUNTIME
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
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

#ifdef AIX_HAS_CUDA_RUNTIME
    const auto is_pinned_pointer = [](const void* pointer) {
        cudaPointerAttributes attributes{};
        const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
        if (status != cudaSuccess) {
            // cudaPointerGetAttributes reports an error for ordinary pageable memory.
            cudaGetLastError();
            return false;
        }
        return attributes.type == cudaMemoryTypeHost;
    };
    if (device_id_ > -1) {
        input_is_pinned_ = is_pinned_pointer(inputData);
        output_is_pinned_ = is_pinned_pointer(outputData);
    }
#endif

    at::IntArrayRef input_sizes = input_shape;
    input_ = torch::from_blob(
        static_cast<void*>(inputData), input_sizes, options.pinned_memory(input_is_pinned_));

    at::IntArrayRef output_sizes = output_shape;
    output_ = torch::from_blob(
        static_cast<void*>(outputData), output_sizes, options.pinned_memory(output_is_pinned_));
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
        const char* async_pipeline_setting = std::getenv("AIX_ASYNC_GPU_PIPELINE");
        const bool use_async_pipeline =
            async_pipeline_setting == nullptr || std::string(async_pipeline_setting) != "0";
        if (use_async_pipeline)
        {
            constexpr int pipeline_depth = 2;
            const char* timeline_directory = std::getenv("AIX_P2P_TIMELINE_DIR");
            const bool timeline_enabled = timeline_directory != nullptr && timeline_directory[0] != '\0';
            const bool timing_enabled = diagnostics_enabled || timeline_enabled;
            struct AsyncBatch {
                torch::Tensor input_cpu;
                torch::Tensor input_gpu;
                torch::Tensor output_gpu;
                torch::Tensor output_cpu;
                int64_t output_start = 0;
                int64_t output_count = 0;
                bool output_in_controller_buffer = false;
                double timeline_anchor_s = 0.0;
                at::cuda::CUDAEvent input_ready;
                at::cuda::CUDAEvent compute_done;
                at::cuda::CUDAEvent output_ready;
                at::cuda::CUDAEvent h2d_start{cudaEventDefault};
                at::cuda::CUDAEvent h2d_end{cudaEventDefault};
                at::cuda::CUDAEvent forward_start{cudaEventDefault};
                at::cuda::CUDAEvent forward_end{cudaEventDefault};
                at::cuda::CUDAEvent d2h_start{cudaEventDefault};
                at::cuda::CUDAEvent d2h_end{cudaEventDefault};
            };

            std::array<AsyncBatch, pipeline_depth> batches;
            const auto device = torch::Device(torch::kCUDA, device_id_);
            const auto h2d_stream = c10::cuda::getStreamFromPool(false, device_id_);
            const auto compute_stream = c10::cuda::getStreamFromPool(false, device_id_);
            const auto d2h_stream = c10::cuda::getStreamFromPool(false, device_id_);

            // A slot is published before reuse so its pinned output buffer and GPU tensors
            // remain valid until both the device copy and the CPU copy have completed.
            const auto publish = [&](AsyncBatch& batch) {
                batch.output_ready.synchronize();
                if (!batch.output_in_controller_buffer) {
                    if (timeline_enabled) {
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            aixelerator_service::utils::p2pTimelineTime(),
                            "torch_controller_output_copy_start", batch.output_start, batch.output_count);
                    }
                    output_.narrow(0, batch.output_start, batch.output_count).copy_(batch.output_cpu);
                    if (timeline_enabled) {
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            aixelerator_service::utils::p2pTimelineTime(),
                            "torch_controller_output_copy_end", batch.output_start, batch.output_count);
                    }
                }
                if (timing_enabled) {
                    const float h2d_ms = batch.h2d_start.elapsed_time(batch.h2d_end);
                    const float forward_ms = batch.forward_start.elapsed_time(batch.forward_end);
                    const float d2h_ms = batch.d2h_start.elapsed_time(batch.d2h_end);
                    if (diagnostics_enabled) {
                        last_timing_.h2d_gpu_ms += h2d_ms;
                        last_timing_.forward_gpu_ms += forward_ms;
                        last_timing_.d2h_gpu_ms += d2h_ms;
                        ++last_timing_.device_batches;
                    }
                    if (timeline_enabled) {
                        // CUDA events share a device timebase. Anchor the first H2D event to
                        // the corrected MPI clock, then preserve actual inter-stream overlap.
                        const double milliseconds_to_seconds = 1.0e-3;
                        const auto phase_time = [&](const at::cuda::CUDAEvent& event) {
                            return batch.timeline_anchor_s + milliseconds_to_seconds *
                                batch.h2d_start.elapsed_time(event);
                        };
                        const double h2d_start_s = batch.timeline_anchor_s;
                        const double h2d_end_s = h2d_start_s + milliseconds_to_seconds * h2d_ms;
                        const double forward_start_s = phase_time(batch.forward_start);
                        const double forward_end_s = forward_start_s + milliseconds_to_seconds * forward_ms;
                        const double d2h_start_s = phase_time(batch.d2h_start);
                        const double d2h_end_s = d2h_start_s + milliseconds_to_seconds * d2h_ms;
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            h2d_start_s, "torch_h2d_start", batch.output_start, batch.output_count);
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            h2d_end_s, "torch_h2d_end", batch.output_start, batch.output_count);
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            forward_start_s, "torch_forward_start", batch.output_start, batch.output_count);
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            forward_end_s, "torch_forward_end", batch.output_start, batch.output_count);
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            d2h_start_s, "torch_d2h_start", batch.output_start, batch.output_count);
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            d2h_end_s, "torch_d2h_end", batch.output_start, batch.output_count);
                    }
                }
            };

            for (int i = 0; i < num_batches; ++i)
            {
                AsyncBatch& batch = batches[static_cast<size_t>(i) % pipeline_depth];
                if (i >= pipeline_depth) {
                    publish(batch);
                }

                const int64_t sample_start = static_cast<int64_t>(batchsize_) * i;
                const int64_t sample_count = std::min<int64_t>(batchsize_, batch_dim - sample_start);
                input_batch_ = input_.narrow(0, sample_start, sample_count);
                batch.output_start = sample_start;
                batch.output_count = sample_count;

                if (input_is_pinned_) {
                    batch.input_cpu = input_batch_;
                }
                else {
                    // Direct Torch users may supply pageable memory; retain the safe staging fallback.
                    if (timeline_enabled) {
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            aixelerator_service::utils::p2pTimelineTime(),
                            "torch_input_stage_alloc_start", sample_start, sample_count);
                    }
                    batch.input_cpu = torch::empty_like(
                        input_batch_, input_batch_.options().device(torch::kCPU).pinned_memory(true));
                    if (timeline_enabled) {
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            aixelerator_service::utils::p2pTimelineTime(),
                            "torch_input_stage_alloc_end", sample_start, sample_count);
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            aixelerator_service::utils::p2pTimelineTime(),
                            "torch_input_stage_copy_start", sample_start, sample_count);
                    }
                    batch.input_cpu.copy_(input_batch_);
                    if (timeline_enabled) {
                        aixelerator_service::utils::writeP2PTorchForwardEventAt(
                            aixelerator_service::utils::p2pTimelineTime(),
                            "torch_input_stage_copy_end", sample_start, sample_count);
                    }
                }

                {
                    c10::cuda::CUDAStreamGuard h2d_guard(h2d_stream);
                    if (batch.compute_done.isCreated()) {
                        batch.compute_done.block(h2d_stream);
                    }
                    if (timing_enabled) {
                        batch.timeline_anchor_s = aixelerator_service::utils::p2pTimelineTime();
                        batch.h2d_start.record(h2d_stream);
                    }
                    batch.input_gpu = batch.input_cpu.to(
                        batch.input_cpu.options().device(device).pinned_memory(false), true);
                    if (timing_enabled) batch.h2d_end.record(h2d_stream);
                    batch.input_ready.record(h2d_stream);
                }

                {
                    c10::cuda::CUDAStreamGuard compute_guard(compute_stream);
                    batch.input_ready.block(compute_stream);
                    std::vector<torch::jit::IValue> inputs = {batch.input_gpu};
                    if (timing_enabled) batch.forward_start.record(compute_stream);
                    batch.output_gpu = torch_model_.forward(inputs).toTensor();
                    if (timing_enabled) batch.forward_end.record(compute_stream);
                    batch.compute_done.record(compute_stream);
                }

                {
                    c10::cuda::CUDAStreamGuard d2h_guard(d2h_stream);
                    batch.compute_done.block(d2h_stream);
                    batch.output_in_controller_buffer = output_is_pinned_;
                    if (batch.output_in_controller_buffer) {
                        batch.output_cpu = output_.narrow(0, batch.output_start, batch.output_count);
                    }
                    else {
                        if (timeline_enabled) {
                            aixelerator_service::utils::writeP2PTorchForwardEventAt(
                                aixelerator_service::utils::p2pTimelineTime(),
                                "torch_output_stage_alloc_start", batch.output_start, batch.output_count);
                        }
                        batch.output_cpu = torch::empty_like(
                            batch.output_gpu,
                            batch.output_gpu.options().device(torch::kCPU).pinned_memory(true));
                        if (timeline_enabled) {
                            aixelerator_service::utils::writeP2PTorchForwardEventAt(
                                aixelerator_service::utils::p2pTimelineTime(),
                                "torch_output_stage_alloc_end", batch.output_start, batch.output_count);
                        }
                    }
                    if (timing_enabled) batch.d2h_start.record(d2h_stream);
                    batch.output_cpu.copy_(batch.output_gpu, true);
                    if (timing_enabled) batch.d2h_end.record(d2h_stream);
                    batch.output_ready.record(d2h_stream);
                }
            }

            const int first_unpublished = std::max(0, num_batches - pipeline_depth);
            for (int i = first_unpublished; i < num_batches; ++i) {
                publish(batches[static_cast<size_t>(i) % pipeline_depth]);
            }
        }
        else
#endif
        {
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
            const int64_t timeline_sample_start = aixelerator_service::utils::p2pTimelineEnvironmentInt64(
                "AIX_P2P_TIMELINE_SAMPLE_BASE", 0) + static_cast<int64_t>(batchsize_) * i;
#ifdef SCOREP
                SCOREP_USER_REGION_BEGIN(h2dCopyHandle, "h2d_copy", SCOREP_USER_REGION_TYPE_COMMON)
            #endif
#ifdef AIX_HAS_CUDA_RUNTIME
            if (diagnostics_enabled) cudaEventRecord(h2d_start);
#endif
            aixelerator_service::utils::writeP2PTorchForwardEvent(
                "torch_h2d_start", timeline_sample_start, input_batch_.size(0));
            input_gpu_ = input_batch_.to(torch::Device(torch::kCUDA, device_id_));
            aixelerator_service::utils::writeP2PTorchForwardEvent(
                "torch_h2d_end", timeline_sample_start, input_batch_.size(0));
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
            aixelerator_service::utils::writeP2PTorchForwardEvent(
                "torch_forward_start", timeline_sample_start, input_batch_.size(0));
            try { output_gpu_ = torch_model_.forward(inputs).toTensor(); } catch (const std::exception& e) { std::cerr << "INPUT SHAPE: "; for(int k=0; k<input_gpu_.dim(); ++k) std::cerr << input_gpu_.size(k) << " "; std::cerr << "\nException: " << e.what() << "\n"; throw; }
            aixelerator_service::utils::writeP2PTorchForwardEvent(
                "torch_forward_end", timeline_sample_start, input_batch_.size(0));
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
            aixelerator_service::utils::writeP2PTorchForwardEvent(
                "torch_d2h_start", timeline_sample_start, input_batch_.size(0));
            output_.slice(0, batchsize_*i, batchsize_*(i+1)) = output_gpu_.to(torch::kCPU);
            aixelerator_service::utils::writeP2PTorchForwardEvent(
                "torch_d2h_end", timeline_sample_start, input_batch_.size(0));
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

template<typename T>
void TorchInference<T>::inferenceRange(int64_t start_sample, int64_t sample_count)
{
    if (start_sample < 0 || sample_count < 0 || start_sample + sample_count > input_.size(0)) {
        throw std::out_of_range("Torch range inference is outside the configured input tensor.");
    }
    if (sample_count == 0) {
        return;
    }

    const auto full_input = input_;
    const auto full_output = output_;
    input_ = input_.narrow(0, start_sample, sample_count);
    output_ = output_.narrow(0, start_sample, sample_count);
    inference();
    input_ = full_input;
    output_ = full_output;
}


template class TorchInference<float>;
template class TorchInference<double>;
