#include "communicationStrategy/collectiveCommunication.h"

#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#ifdef SCOREP
#include <scorep/SCOREP_User.h>
SCOREP_USER_REGION_DEFINE(pipelinedExchangeHandle)
#endif


template<typename T>
MPI_Datatype getTypeFromTemplate()
{
    if (std::is_same<T, float>::value)
    {
        return MPI_FLOAT;
    }

    if (std::is_same<T, double>::value)
    {
        return MPI_DOUBLE;
    }

    std::cerr << "ERROR: Could not get MPI_Datatype from template - defaulting to MPI_FLOAT!" << std::endl;
    // TODO(fabian): find a nicer way to handle error. Maybe use std::optional as return type?
    return MPI_FLOAT;
}


template<typename T>
CollectiveCommunication<T>::CollectiveCommunication(
    std::vector<int64_t> input_shape, T* input_data, 
    std::vector<int64_t> output_shape, T* output_data, 
    bool is_device_controller, MPI_Comm work_group_comm)
{

    is_device_controller_ = is_device_controller;
    work_group_comm_ = work_group_comm;
    MPI_Comm_dup(work_group_comm_, &pipelined_comm_);
    dtype_ = getTypeFromTemplate<T>();
    MPI_Comm_size(work_group_comm_, &workgroup_size_);

    int workgroup_rank = -1;
    MPI_Comm_rank(work_group_comm_, &workgroup_rank);
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(work_group_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int node_rank = -1;
    MPI_Comm_rank(node_comm, &node_rank);
    const int node_leader_candidate = node_rank == 0 ? workgroup_rank : -1;
    int node_leader = -1;
    MPI_Allreduce(&node_leader_candidate, &node_leader, 1, MPI_INT, MPI_MAX, node_comm);
    MPI_Comm_free(&node_comm);
    node_leaders_.resize(workgroup_size_);
    MPI_Allgather(&node_leader, 1, MPI_INT, node_leaders_.data(), 1, MPI_INT, work_group_comm_);

    int input_sendcount = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    int output_sendcount = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    setInputData(input_sendcount, input_data);
    setOutputData(output_sendcount, output_data);

    // determine number of samples over all processes
    int batch_dim = input_shape[0];
    int total_batch_dim = 0;
    MPI_Reduce(&batch_dim, &total_batch_dim, 1, MPI_INT, MPI_SUM, 0, work_group_comm_);
    this->total_input_samples_ = total_batch_dim;

    this->input_shape_controller_ = input_shape;
    this->input_shape_controller_[0] = total_batch_dim;
    this->output_shape_controller_ = output_shape;
    this->output_shape_controller_[0] = total_batch_dim;

    if( is_device_controller_ )
    {
        input_recvcounts_.resize(workgroup_size_, -1);
        input_displs_.resize(workgroup_size_, -1);

        output_recvcounts_.resize(workgroup_size_, -1);
        output_displs_.resize(workgroup_size_, -1);
    }
    else
    {
        input_recvcounts_.resize(0);
        input_displs_.resize(0);

        output_recvcounts_.resize(0);
        output_displs_.resize(0);
    }
    
    MPI_Gather(&input_sendcount_, 1, MPI_INT, input_recvcounts_.data(), 1, MPI_INT, 0, work_group_comm_);
    MPI_Gather(&output_sendcount_, 1, MPI_INT, output_recvcounts_.data(), 1, MPI_INT, 0, work_group_comm_);
    
    if( is_device_controller_ )
    {
        input_displs_[0] = 0;
        output_displs_[0] = 0;
        for(int i = 1; i < workgroup_size_; i++)
        {
            input_displs_[i] = input_displs_[i-1] + input_recvcounts_[i-1];
            output_displs_[i] = output_displs_[i-1] + output_recvcounts_[i-1];
        }  
        this->total_input_count_ = std::accumulate(input_recvcounts_.begin(), input_recvcounts_.end(), (int64_t)0);
        this->total_output_count_ = std::accumulate(output_recvcounts_.begin(), output_recvcounts_.end(), (int64_t)0);

        this->input_data_controller_ = new T[this->total_input_count_];
        this->output_data_controller_ = new T[this->total_output_count_];
    }
}

template<typename T>
CollectiveCommunication<T>::~CollectiveCommunication()
{
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized && pipelined_comm_ != MPI_COMM_NULL)
    {
        MPI_Comm_free(&pipelined_comm_);
    }
    if( is_device_controller_ )
    {
        delete[] this->input_data_controller_;
        delete[] this->output_data_controller_;
    }
}

template<typename T>
void CollectiveCommunication<T>::setInputData(int input_sendcount, T* input_data)
{
    input_sendcount_ = input_sendcount;
    input_data_worker_ = input_data;
}

template<typename T>
void CollectiveCommunication<T>::setOutputData(int output_sendcount, T* output_data)
{
    output_sendcount_ = output_sendcount;
    output_data_worker_ = output_data;
}

template<typename T>
void CollectiveCommunication<T>::gatherInputData()
{
    const char* dump_dir = std::getenv("DUMP_TENSOR_DIR");
    if (dump_dir) {
        int rank; MPI_Comm_rank(work_group_comm_, &rank);
        std::ostringstream tag;
        tag << dump_dir << "/aix_gather_src_wg" << rank << "_ptr" << (void*)input_data_worker_ << "_" << input_sendcount_ << "elems";
        std::ofstream(tag.str() + ".bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(input_data_worker_), input_sendcount_ * sizeof(T));
    }
    MPI_Gatherv(input_data_worker_, input_sendcount_, dtype_, this->input_data_controller_, input_recvcounts_.data(), input_displs_.data(), dtype_, 0, work_group_comm_);
}

template<typename T>
void CollectiveCommunication<T>::scatterOutputData()
{
    MPI_Scatterv(this->output_data_controller_, output_recvcounts_.data(), output_displs_.data(), dtype_, output_data_worker_, output_sendcount_, dtype_, 0, work_group_comm_);
}

template<typename T>
void CollectiveCommunication<T>::pipelinedExchange(const std::function<void(int64_t, int64_t)>& infer_range)
{
    constexpr int credit_tag = 17001;
    constexpr int input_tag = 17002;
    constexpr int output_tag = 17003;

#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN(pipelinedExchangeHandle, "p2p_pipelined_exchange", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif

    if (!is_device_controller_)
    {
        int credit = 0;
        MPI_Recv(&credit, 1, MPI_INT, 0, credit_tag, pipelined_comm_, MPI_STATUS_IGNORE);

        MPI_Request input_request = MPI_REQUEST_NULL;
        MPI_Isend(input_data_worker_, input_sendcount_, dtype_, 0, input_tag, pipelined_comm_, &input_request);
        MPI_Wait(&input_request, MPI_STATUS_IGNORE);
        MPI_Recv(output_data_worker_, output_sendcount_, dtype_, 0, output_tag, pipelined_comm_, MPI_STATUS_IGNORE);
#ifdef SCOREP
        SCOREP_USER_REGION_END(pipelinedExchangeHandle)
#endif
        return;
    }

    if (workgroup_size_ == 0 || input_recvcounts_.size() != static_cast<size_t>(workgroup_size_))
    {
        throw std::runtime_error("Pipelined communication requires controller receive metadata.");
    }

    const auto elements_per_sample = [](const std::vector<int64_t>& shape) {
        if (shape.empty()) {
            throw std::runtime_error("Pipelined communication requires a batched tensor shape.");
        }
        return std::accumulate(std::next(shape.begin()), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    };
    const int64_t input_elements_per_sample = elements_per_sample(this->input_shape_controller_);
    const int64_t output_elements_per_sample = elements_per_sample(this->output_shape_controller_);
    if (input_elements_per_sample <= 0 || output_elements_per_sample <= 0)
    {
        throw std::runtime_error("Pipelined communication requires non-empty per-sample tensors.");
    }

    for (int rank = 0; rank < workgroup_size_; ++rank)
    {
        if (input_recvcounts_[rank] % input_elements_per_sample != 0 ||
            input_displs_[rank] % input_elements_per_sample != 0 ||
            output_recvcounts_[rank] % output_elements_per_sample != 0 ||
            output_displs_[rank] % output_elements_per_sample != 0)
        {
            throw std::runtime_error("Pipelined communication requires rank segments aligned to whole samples.");
        }
    }

    // Rank zero owns the controller-local segment. All other segments are populated
    // by posted receives, so completed ranks always describe valid tensor ranges.
    std::copy_n(input_data_worker_, input_sendcount_, this->input_data_controller_ + input_displs_[0]);

    std::vector<MPI_Request> input_requests(workgroup_size_, MPI_REQUEST_NULL);
    std::vector<MPI_Request> output_requests(workgroup_size_, MPI_REQUEST_NULL);
    std::vector<MPI_Request> credit_requests(workgroup_size_, MPI_REQUEST_NULL);
    std::vector<int> credit_values(workgroup_size_, 1);
    for (int rank = 1; rank < workgroup_size_; ++rank)
    {
        MPI_Irecv(this->input_data_controller_ + input_displs_[rank], input_recvcounts_[rank], dtype_, rank,
                  input_tag, pipelined_comm_, &input_requests[rank]);
    }

    std::vector<std::vector<int>> tiers(2);
    const int controller_node_leader = node_leaders_[0];
    for (int rank = 1; rank < workgroup_size_; ++rank)
    {
        const int tier = node_leaders_[rank] == controller_node_leader ? 0 : 1;
        tiers[tier].push_back(rank);
    }

    std::array<int, 2> next_rank = {0, 0};
    std::array<int, 2> level = {1, 1};
    std::array<int, 2> expected_in_level = {0, 0};
    std::array<int, 2> completed_in_level = {0, 0};
    const auto issue_level = [&](int tier) {
        const int remaining = static_cast<int>(tiers[tier].size()) - next_rank[tier];
        const int grant_count = std::min(level[tier], remaining);
        expected_in_level[tier] = grant_count;
        completed_in_level[tier] = 0;
        for (int i = 0; i < grant_count; ++i)
        {
            const int rank = tiers[tier][next_rank[tier]++];
            MPI_Isend(&credit_values[rank], 1, MPI_INT, rank, credit_tag, pipelined_comm_, &credit_requests[rank]);
        }
    };
    for (int tier = 0; tier < 2; ++tier)
    {
        if (!tiers[tier].empty())
        {
            issue_level(tier);
        }
    }

    std::vector<bool> received(workgroup_size_, false);
    std::vector<bool> inferred(workgroup_size_, false);
    received[0] = true;
    int received_count = 1;
    int inferred_count = 0;

    const auto send_results = [&](int first_rank, int end_rank) {
        for (int rank = first_rank; rank < end_rank; ++rank)
        {
            if (rank == 0)
            {
                std::copy_n(this->output_data_controller_ + output_displs_[rank], output_recvcounts_[rank], output_data_worker_);
            }
            else
            {
                MPI_Isend(this->output_data_controller_ + output_displs_[rank], output_recvcounts_[rank], dtype_, rank,
                          output_tag, pipelined_comm_, &output_requests[rank]);
            }
        }
    };

    const auto infer_largest_ready_range = [&]() -> bool {
        int best_start = -1;
        int best_end = -1;
        for (int rank = 0; rank < workgroup_size_;)
        {
            if (!received[rank] || inferred[rank])
            {
                ++rank;
                continue;
            }
            int end = rank + 1;
            while (end < workgroup_size_ && received[end] && !inferred[end])
            {
                ++end;
            }
            if (best_start < 0 || end - rank > best_end - best_start)
            {
                best_start = rank;
                best_end = end;
            }
            rank = end;
        }
        if (best_start < 0)
        {
            return false;
        }

        const int64_t start_sample = input_displs_[best_start] / input_elements_per_sample;
        const int64_t end_sample = (input_displs_[best_end - 1] + input_recvcounts_[best_end - 1]) / input_elements_per_sample;
        if (end_sample > start_sample)
        {
            infer_range(start_sample, end_sample - start_sample);
        }
        for (int rank = best_start; rank < best_end; ++rank)
        {
            inferred[rank] = true;
            ++inferred_count;
        }
        send_results(best_start, best_end);
        return true;
    };

    std::vector<int> completed_indices(workgroup_size_);
    std::vector<MPI_Status> completed_statuses(workgroup_size_);
    while (inferred_count < workgroup_size_)
    {
        int completed = 0;
        MPI_Testsome(workgroup_size_, input_requests.data(), &completed, completed_indices.data(), completed_statuses.data());
        if (completed != MPI_UNDEFINED)
        {
            for (int i = 0; i < completed; ++i)
            {
                const int rank = completed_indices[i];
                if (!received[rank])
                {
                    received[rank] = true;
                    ++received_count;
                    const int tier = node_leaders_[rank] == controller_node_leader ? 0 : 1;
                    ++completed_in_level[tier];
                    if (completed_in_level[tier] == expected_in_level[tier] && next_rank[tier] < static_cast<int>(tiers[tier].size()))
                    {
                        level[tier] = std::min(level[tier] * 2, static_cast<int>(tiers[tier].size()) - next_rank[tier]);
                        issue_level(tier);
                    }
                }
            }
        }

        if (infer_largest_ready_range())
        {
            continue;
        }
        if (received_count < workgroup_size_)
        {
            int completed_after_wait = 0;
            MPI_Waitsome(workgroup_size_, input_requests.data(), &completed_after_wait, completed_indices.data(), completed_statuses.data());
            if (completed_after_wait == MPI_UNDEFINED)
            {
                throw std::runtime_error("Pipelined input requests completed before all rank segments were observed.");
            }
            for (int i = 0; i < completed_after_wait; ++i)
            {
                const int rank = completed_indices[i];
                if (!received[rank])
                {
                    received[rank] = true;
                    ++received_count;
                    const int tier = node_leaders_[rank] == controller_node_leader ? 0 : 1;
                    ++completed_in_level[tier];
                    if (completed_in_level[tier] == expected_in_level[tier] && next_rank[tier] < static_cast<int>(tiers[tier].size()))
                    {
                        level[tier] = std::min(level[tier] * 2, static_cast<int>(tiers[tier].size()) - next_rank[tier]);
                        issue_level(tier);
                    }
                }
            }
        }
    }

    MPI_Waitall(workgroup_size_, output_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(workgroup_size_, credit_requests.data(), MPI_STATUSES_IGNORE);
#ifdef SCOREP
    SCOREP_USER_REGION_END(pipelinedExchangeHandle)
#endif
}


template class CollectiveCommunication<float>;
template class CollectiveCommunication<double>;
