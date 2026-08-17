#include "aixeleratorService/aixeleratorService_interfaceC.h"
#include "aixeleratorService/aixeleratorService.h"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

    AIxeleratorServiceHandle createAIxeleratorServiceDouble_F(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize, int app_comm
    ){
        return createAIxeleratorServiceDoubleWithMode_F(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize, app_comm, 0);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloat_F(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize, int app_comm
    ){
        return createAIxeleratorServiceFloatWithMode_F(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize, app_comm, 0);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceDoubleWithMode_F(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize, int app_comm, int communication_mode
    ){
        return createAIxeleratorServiceDoubleWithMode(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize, MPI_Comm_f2c(app_comm), communication_mode);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloatWithMode_F(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize, int app_comm, int communication_mode
    ){
        return createAIxeleratorServiceFloatWithMode(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize, MPI_Comm_f2c(app_comm), communication_mode);
    }


    AIxeleratorServiceHandle createAIxeleratorServiceDouble(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize, MPI_Comm app_comm
    ){
        return createAIxeleratorServiceDoubleWithMode(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize, app_comm, 0);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloat(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize, MPI_Comm app_comm
    ){
        return createAIxeleratorServiceFloatWithMode(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize, app_comm, 0);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceDoubleWithMode(
        char* model_file,
        int64_t* input_shape, int num_input_dims, double* input_data,
        int64_t* output_shape, int num_output_dims, double* output_data,
        int batchsize, MPI_Comm app_comm, int communication_mode
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);
        auto mode = communication_mode == 1 ? CommunicationMode::Pipelined : CommunicationMode::Collective;

        return (AIxeleratorService<double>*) new AIxeleratorService<double>(
            model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize, app_comm, false, std::nullopt, mode);
    }

    AIxeleratorServiceHandle createAIxeleratorServiceFloatWithMode(
        char* model_file,
        int64_t* input_shape, int num_input_dims, float* input_data,
        int64_t* output_shape, int num_output_dims, float* output_data,
        int batchsize, MPI_Comm app_comm, int communication_mode
    ){
        std::string model_file_str(model_file);
        std::vector<int64_t> input_shape_vec(input_shape, input_shape + num_input_dims);
        std::vector<int64_t> output_shape_vec(output_shape, output_shape + num_output_dims);
        auto mode = communication_mode == 1 ? CommunicationMode::Pipelined : CommunicationMode::Collective;

        return (AIxeleratorService<float>*) new AIxeleratorService<float>(
            model_file_str, input_shape_vec, input_data, output_shape_vec, output_data, batchsize, app_comm, false, std::nullopt, mode);
    }

    void deleteAIxeleratorServiceDouble(AIxeleratorServiceHandle aixelerator)
    {
        delete (AIxeleratorService<double>*) aixelerator;
    }

    void deleteAIxeleratorServiceFloat(AIxeleratorServiceHandle aixelerator)
    {
        delete (AIxeleratorService<float>*) aixelerator;
    }

    void inferenceAIxeleratorServiceDouble(AIxeleratorServiceHandle aixelerator)
    {
        ((AIxeleratorService<double>*) aixelerator)->inference();    
    }

    void inferenceAIxeleratorServiceFloat(AIxeleratorServiceHandle aixelerator)
    {
        ((AIxeleratorService<float>*) aixelerator)->inference();    
    }

    void setAIxeleratorServiceDebugTag(AIxeleratorServiceHandle aixelerator, char* debug_tag)
    {
        std::string tag(debug_tag);
        ((AIxeleratorService<float>*) aixelerator)->setDebugTag(tag);    
    }

#ifdef __cplusplus
}
#endif
