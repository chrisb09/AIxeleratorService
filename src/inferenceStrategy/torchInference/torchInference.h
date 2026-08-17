#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_TORCHINFERENCE_H_

#include "inferenceStrategy/inferenceStrategy.h"
#include <torch/script.h>

#include <memory>

template<typename T>
class TorchInference : public InferenceStrategy<T>
{
    public:
        TorchInference();
        ~TorchInference();

        void init(
            int batchsize, 
            int device_id, 
            std::string model_file_name, 
            std::vector<int64_t>& input_shape, T* inputData, 
            std::vector<int64_t>& output_shape, T* outputData
        ) override;
        void setControllerDirectBuffers(T* input_worker, T* output_worker) override;
        //void setInput() override;
        //void setOutput() override;
        void inference() override;
        void inferenceRange(int64_t start_sample, int64_t sample_count) override;
        bool supportsRangePipeline() const override;
        bool canSubmitRangePipeline() const override;
        int64_t maxRangePipelineSamples() const override;
        uint64_t submitRangePipeline(int64_t start_sample, int64_t sample_count,
                                     int first_rank, int end_rank) override;
        bool rangePipelineComplete(uint64_t request_id) override;
        void releaseRangePipeline(uint64_t request_id) override;
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
        T* controller_input_worker_ = nullptr;
        T* controller_output_worker_ = nullptr;
        InferenceTiming last_timing_;
        struct RangePipeline;
        std::unique_ptr<RangePipeline> range_pipeline_;
};


#endif
