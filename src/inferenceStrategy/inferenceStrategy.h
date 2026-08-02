#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_H_

#include <string>
#include <vector>

#include <mpi.h>

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
        virtual void inference() = 0;
        virtual InferenceTiming getLastTiming() const { return {}; }

        std::string debug_tag_;
        void setDebugTag(std::string tag){ debug_tag_ = tag; }
        std::string getDebugTag(){ return debug_tag_; }

        MPI_Comm comm_;
        void setCommunicator(MPI_Comm comm){ comm_ = comm; }
};

#endif
