#ifndef AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_H_
#define AIXELERATORSERVICE_COMMUNICATIONSTRATEGY_H_

#include <vector>
#include <cstdint>
#include <functional>
#include <stdexcept>

struct PipelinedInferenceExecutor
{
    std::function<void(int64_t, int64_t)> infer_range;
    std::function<bool()> can_submit;
    std::function<int64_t()> max_samples;
    std::function<uint64_t(int64_t, int64_t, int, int)> submit;
    std::function<bool(uint64_t)> complete;
    std::function<void(uint64_t)> release;

    bool supportsRangePipeline() const
    {
        return can_submit && max_samples && submit && complete && release;
    }
};

template<typename T>
class CommunicationStrategy
{
    public:
        virtual ~CommunicationStrategy() = default;

        virtual void gatherInputData() = 0;
        virtual void scatterOutputData() = 0;
        virtual void pipelinedExchange(const PipelinedInferenceExecutor&)
        {
            throw std::runtime_error("The selected communication strategy does not support pipelined exchange.");
        }

        T* getInputDataController(){ return input_data_controller_; }
        T* getOutputDataController(){ return output_data_controller_; }
    
        std::vector<int64_t> getInputShapeController(){return input_shape_controller_;}
        std::vector<int64_t> getOutputShapeController(){return output_shape_controller_;}

        int getTotalInputCount(){return total_input_count_;}
        int getTotalOutputCount(){return total_output_count_;}

        int getTotalInputSamples(){return total_input_samples_;}

    protected:
        T* input_data_controller_;
        T* output_data_controller_;

        std::vector<int64_t> input_shape_controller_;
        std::vector<int64_t> output_shape_controller_; 

        int64_t total_input_count_;
        int64_t total_output_count_;  

        int64_t total_input_samples_;     
};

#endif
