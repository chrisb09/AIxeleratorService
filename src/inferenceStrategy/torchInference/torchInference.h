#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"
#include <torch/script.h>

template<typename T>
class TorchInference : public InferenceStrategy<T>
{
    public:
        TorchInference() = default;
        ~TorchInference() = default;

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, T* inputData, 
            std::vector<int64_t>& output_shape, T* outputData
        ) override;
        //void setInput() override;
        //void setOutput() override;
        void inference() override;
        void inferenceRange(int64_t start_sample, int64_t sample_count) override;
        InferenceTiming getLastTiming() const override { return last_timing_; }

    private:
        std::string model_file_name_;
        torch::jit::script::Module torch_model_;
        int batchsize_;
        int device_id_;
        torch::Tensor input_;
        torch::Tensor input_batch_;
        torch::Tensor input_gpu_;
        torch::Tensor output_;
        torch::Tensor output_batch_;
        torch::Tensor output_gpu_;
        bool input_is_pinned_ = false;
        bool output_is_pinned_ = false;
        InferenceTiming last_timing_;
};


#endif
