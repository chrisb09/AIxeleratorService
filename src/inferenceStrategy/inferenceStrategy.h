#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_H_

#include <string>
#include <vector>
#include <cstdint>
#include <limits>

#include <mpi.h>
#include <stdexcept>

struct InferenceTiming {
    double h2d_gpu_ms = 0.0;
    double forward_gpu_ms = 0.0;
    double d2h_gpu_ms = 0.0;
    int device_batches = 0;
};

template<typename T>
class InferenceStrategy
{
    public:
        InferenceStrategy() = default;
        InferenceStrategy(std::string& model_file_name);
        virtual ~InferenceStrategy() = default;

        //virtual void setInput(std::vector<int> input_shape, double* input_data) = 0;
        //virtual void setOutput(std::vector<int> output_shape, double* output_data) = 0;
        virtual void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, T* inputData, 
            std::vector<int64_t>& output_shape, T* outputData
        ) = 0;
        virtual void setControllerDirectBuffers(T* input_worker, T* output_worker)
        {
            (void)input_worker;
            (void)output_worker;
        }
        virtual void inference() = 0;
        virtual void inferenceRange(int64_t, int64_t)
        {
            throw std::runtime_error("The selected inference backend does not support range inference.");
        }
        virtual bool supportsRangePipeline() const { return false; }
        virtual bool canSubmitRangePipeline() const { return false; }
        virtual int64_t maxRangePipelineSamples() const { return std::numeric_limits<int64_t>::max(); }
        virtual uint64_t submitRangePipeline(int64_t, int64_t, int, int)
        {
            throw std::runtime_error("The selected inference backend does not support range pipelining.");
        }
        virtual bool rangePipelineComplete(uint64_t)
        {
            throw std::runtime_error("The selected inference backend does not support range pipelining.");
        }
        virtual void releaseRangePipeline(uint64_t)
        {
            throw std::runtime_error("The selected inference backend does not support range pipelining.");
        }
        virtual InferenceTiming getLastTiming() const { return {}; }

        std::string debug_tag_;
        void setDebugTag(std::string tag){ debug_tag_ = tag; }
        std::string getDebugTag(){ return debug_tag_; }

        MPI_Comm comm_;
        void setCommunicator(MPI_Comm comm){ comm_ = comm; }
};

#endif
