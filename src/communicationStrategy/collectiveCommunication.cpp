#include "communicationStrategy/collectiveCommunication.h"

#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <future>
#include <thread>
#include <limits>
#include <stdexcept>

#include "utils/p2pTimeline.h"

#ifdef AIX_HAS_CUDA_RUNTIME
#include <cuda_runtime_api.h>
#endif

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

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_rank(work_group_comm_, &workgroup_rank_);
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(work_group_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int node_rank = -1;
    MPI_Comm_rank(node_comm, &node_rank);
    const int node_leader_candidate = node_rank == 0 ? workgroup_rank_ : -1;
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

#ifdef AIX_HAS_CUDA_RUNTIME
        const auto allocate_pinned = [](T*& buffer, int64_t count, const char* name) {
            const cudaError_t status = cudaHostAlloc(
                reinterpret_cast<void**>(&buffer), static_cast<size_t>(count) * sizeof(T), cudaHostAllocDefault);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Could not allocate pinned controller ") + name + " buffer: " +
                    cudaGetErrorString(status));
                }
        };
        this->input_data_controller_ = nullptr;
        this->output_data_controller_ = nullptr;
        try {
            allocate_pinned(this->input_data_controller_, this->total_input_count_, "input");
            allocate_pinned(this->output_data_controller_, this->total_output_count_, "output");
            controller_buffers_pinned_ = true;
        }
        catch (...) {
            if (this->input_data_controller_ != nullptr) {
                cudaFreeHost(this->input_data_controller_);
                this->input_data_controller_ = nullptr;
            }
            throw;
        }
#else
        this->input_data_controller_ = new T[this->total_input_count_];
        this->output_data_controller_ = new T[this->total_output_count_];
#endif
    }
}

template<typename T>
CollectiveCommunication<T>::~CollectiveCommunication()
{
    aixelerator_service::utils::flushP2PTimelineEvents();
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized && pipelined_comm_ != MPI_COMM_NULL)
    {
        MPI_Comm_free(&pipelined_comm_);
    }
    if( is_device_controller_ )
    {
#ifdef AIX_HAS_CUDA_RUNTIME
        if (controller_buffers_pinned_) {
            cudaFreeHost(this->input_data_controller_);
            cudaFreeHost(this->output_data_controller_);
            return;
        }
#endif
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
    const uint64_t step = pipelined_step_++;
    const int initial_credits = std::max(
        1, aixelerator_service::utils::p2pTimelineEnvironmentInt("AIX_P2P_INITIAL_CREDITS", 4));
    int mpi_thread_level = MPI_THREAD_SINGLE;
    MPI_Query_thread(&mpi_thread_level);
    const char* async_controller_setting = std::getenv("AIX_P2P_ASYNC_CONTROLLER");
    const bool async_controller = mpi_thread_level >= MPI_THREAD_MULTIPLE &&
        (async_controller_setting == nullptr || std::string(async_controller_setting) != "0");

    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, is_device_controller_, "ml_step_start");

#ifdef SCOREP
    SCOREP_USER_REGION_BEGIN(pipelinedExchangeHandle, "p2p_pipelined_exchange", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif

    if (!is_device_controller_)
    {
        int credit = 0;
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, false, "input_credit_wait_start");
        MPI_Recv(&credit, 1, MPI_INT, 0, credit_tag, pipelined_comm_, MPI_STATUS_IGNORE);

        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, false, "input_send_start", 0,
            workgroup_rank_, workgroup_rank_ + 1);
        MPI_Send(input_data_worker_, input_sendcount_, dtype_, 0, input_tag, pipelined_comm_);
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, false, "input_send_complete", 0);
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, false, "result_wait_start", 0);
        MPI_Recv(output_data_worker_, output_sendcount_, dtype_, 0, output_tag, pipelined_comm_, MPI_STATUS_IGNORE);
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, false, "result_received", 0);
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, false, "ml_step_end");
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

    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "input_listen_start");
    std::vector<MPI_Request> input_requests(workgroup_size_, MPI_REQUEST_NULL);
    std::vector<MPI_Request> output_requests(workgroup_size_, MPI_REQUEST_NULL);
    std::vector<MPI_Request> credit_requests(workgroup_size_, MPI_REQUEST_NULL);
    std::vector<int> credit_values(workgroup_size_, 1);
    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "input_post_receives_start");
    for (int rank = 1; rank < workgroup_size_; ++rank)
    {
        MPI_Irecv(this->input_data_controller_ + input_displs_[rank], input_recvcounts_[rank], dtype_, rank,
                  input_tag, pipelined_comm_, &input_requests[rank]);
    }
    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "input_post_receives_end");

    std::vector<std::vector<int>> tiers(2);
    const int controller_node_leader = node_leaders_[0];
    for (int rank = 1; rank < workgroup_size_; ++rank)
    {
        const int tier = node_leaders_[rank] == controller_node_leader ? 0 : 1;
        tiers[tier].push_back(rank);
    }

    std::array<int, 2> next_rank = {0, 0};
    const auto issue_grants = [&](int tier, int requested_grants) {
        const int remaining = static_cast<int>(tiers[tier].size()) - next_rank[tier];
        const int grant_count = std::min(requested_grants, remaining);
        for (int i = 0; i < grant_count; ++i)
        {
            const int rank = tiers[tier][next_rank[tier]++];
            aixelerator_service::utils::writeP2PTimelineEvent(
                step, world_rank_, workgroup_rank_, true, "input_credit_send", rank);
            MPI_Isend(&credit_values[rank], 1, MPI_INT, rank, credit_tag, pipelined_comm_, &credit_requests[rank]);
        }
    };
    for (int tier = 0; tier < 2; ++tier)
    {
        if (!tiers[tier].empty())
        {
            issue_grants(tier, initial_credits);
        }
    }

    // Rank zero owns the controller-local segment. Post remote receives and issue
    // credits first so remote MPI transfers can progress during this local copy.
    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "controller_input_copy_start");
    std::copy_n(input_data_worker_, input_sendcount_, this->input_data_controller_ + input_displs_[0]);
    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "controller_input_copy_end");

    std::vector<bool> received(workgroup_size_, false);
    std::vector<bool> inferred(workgroup_size_, false);
    std::vector<bool> scheduled(workgroup_size_, false);
    received[0] = true;
    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "input_ready", 0, 0, 1,
        input_displs_[0] / input_elements_per_sample, input_recvcounts_[0] / input_elements_per_sample);
    int received_count = 1;
    int inferred_count = 0;

    const auto send_results = [&](int first_rank, int end_rank) {
        for (int rank = first_rank; rank < end_rank; ++rank)
        {
            if (rank == 0)
            {
                aixelerator_service::utils::writeP2PTimelineEvent(
                    step, world_rank_, workgroup_rank_, true, "result_send_start", rank,
                    rank, rank + 1, output_displs_[rank] / output_elements_per_sample,
                    output_recvcounts_[rank] / output_elements_per_sample);
                aixelerator_service::utils::writeP2PTimelineEvent(
                    step, world_rank_, workgroup_rank_, true, "controller_output_copy_start", rank,
                    rank, rank + 1, output_displs_[rank] / output_elements_per_sample,
                    output_recvcounts_[rank] / output_elements_per_sample);
                std::copy_n(this->output_data_controller_ + output_displs_[rank], output_recvcounts_[rank], output_data_worker_);
                aixelerator_service::utils::writeP2PTimelineEvent(
                    step, world_rank_, workgroup_rank_, true, "controller_output_copy_end", rank,
                    rank, rank + 1, output_displs_[rank] / output_elements_per_sample,
                    output_recvcounts_[rank] / output_elements_per_sample);
            }
            else
            {
                aixelerator_service::utils::writeP2PTimelineEvent(
                    step, world_rank_, workgroup_rank_, true, "result_send_start", rank,
                    rank, rank + 1, output_displs_[rank] / output_elements_per_sample,
                    output_recvcounts_[rank] / output_elements_per_sample);
                MPI_Isend(this->output_data_controller_ + output_displs_[rank], output_recvcounts_[rank], dtype_, rank,
                          output_tag, pipelined_comm_, &output_requests[rank]);
            }
        }
    };

    const auto record_received = [&](int rank) {
        if (received[rank]) {
            return;
        }
        received[rank] = true;
        ++received_count;
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, true, "input_ready", rank,
            rank, rank + 1, input_displs_[rank] / input_elements_per_sample,
            input_recvcounts_[rank] / input_elements_per_sample);
        const int tier = node_leaders_[rank] == controller_node_leader ? 0 : 1;
        if (next_rank[tier] < static_cast<int>(tiers[tier].size())) {
            issue_grants(tier, 2);
        }
    };

    bool inference_in_flight = false;
    int in_flight_start_rank = -1;
    int in_flight_end_rank = -1;
    int64_t in_flight_start_sample = 0;
    int64_t in_flight_sample_count = 0;
    std::future<void> inference_future;

    const auto finish_inference = [&]() {
        if (async_controller) {
            inference_future.get();
        }
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, true, "range_inference_end", -1,
            in_flight_start_rank, in_flight_end_rank, in_flight_start_sample, in_flight_sample_count);
        for (int rank = in_flight_start_rank; rank < in_flight_end_rank; ++rank) {
            scheduled[rank] = false;
            inferred[rank] = true;
            ++inferred_count;
        }
        send_results(in_flight_start_rank, in_flight_end_rank);
        inference_in_flight = false;
    };

    const auto start_largest_ready_range = [&]() -> bool {
        int best_start = -1;
        int best_end = -1;
        for (int rank = 0; rank < workgroup_size_;)
        {
            if (!received[rank] || inferred[rank] || scheduled[rank])
            {
                ++rank;
                continue;
            }
            int end = rank + 1;
            while (end < workgroup_size_ && received[end] && !inferred[end] && !scheduled[end])
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
        if (end_sample <= start_sample) {
            return false;
        }
        const auto set_context = [](const char* name, const std::string& value) {
            setenv(name, value.c_str(), 1);
        };
        set_context("AIX_P2P_TIMELINE_WORLD_RANK", std::to_string(world_rank_));
        set_context("AIX_P2P_TIMELINE_WORKGROUP_RANK", std::to_string(workgroup_rank_));
        set_context("AIX_P2P_TIMELINE_STEP", std::to_string(step));
        set_context("AIX_P2P_TIMELINE_RANGE_FIRST_RANK", std::to_string(best_start));
        set_context("AIX_P2P_TIMELINE_RANGE_END_RANK", std::to_string(best_end));
        set_context("AIX_P2P_TIMELINE_SAMPLE_BASE", std::to_string(start_sample));
        for (int rank = best_start; rank < best_end; ++rank) {
            scheduled[rank] = true;
        }
        in_flight_start_rank = best_start;
        in_flight_end_rank = best_end;
        in_flight_start_sample = start_sample;
        in_flight_sample_count = end_sample - start_sample;
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, true, "range_inference_start", -1,
            best_start, best_end, start_sample, end_sample - start_sample);
        if (async_controller) {
            inference_in_flight = true;
            const int task_start_rank = in_flight_start_rank;
            const int task_end_rank = in_flight_end_rank;
            const int64_t task_start_sample = in_flight_start_sample;
            const int64_t task_sample_count = in_flight_sample_count;
            inference_future = std::async(std::launch::async, [&, task_start_rank, task_end_rank,
                                                                 task_start_sample, task_sample_count]() {
                aixelerator_service::utils::P2PTimelineThreadContextGuard context(
                    world_rank_, workgroup_rank_, true, task_start_rank, task_end_rank);
                infer_range(task_start_sample, task_sample_count);
            });
        }
        else {
            infer_range(in_flight_start_sample, in_flight_sample_count);
            finish_inference();
        }
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
                record_received(completed_indices[i]);
            }
        }

        if (inference_in_flight &&
            inference_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            finish_inference();
            continue;
        }
        if (!inference_in_flight && start_largest_ready_range())
        {
            continue;
        }
        if (received_count < workgroup_size_)
        {
            if (inference_in_flight) {
                // Keep driving MPI while the GPU worker finishes. A blocking Waitsome here
                // delays publishing completed GPU work until an unrelated receive arrives.
                std::this_thread::yield();
                continue;
            }
            int completed_after_wait = 0;
            MPI_Waitsome(workgroup_size_, input_requests.data(), &completed_after_wait, completed_indices.data(), completed_statuses.data());
            if (completed_after_wait == MPI_UNDEFINED)
            {
                throw std::runtime_error("Pipelined input requests completed before all rank segments were observed.");
            }
            for (int i = 0; i < completed_after_wait; ++i)
            {
                record_received(completed_indices[i]);
            }
        }
        else if (inference_in_flight)
        {
            inference_future.wait();
            finish_inference();
        }
    }

    MPI_Waitall(workgroup_size_, output_requests.data(), MPI_STATUSES_IGNORE);
    for (int rank = 1; rank < workgroup_size_; ++rank)
    {
        aixelerator_service::utils::writeP2PTimelineEvent(
            step, world_rank_, workgroup_rank_, true, "result_send_complete", rank);
    }
    MPI_Waitall(workgroup_size_, credit_requests.data(), MPI_STATUSES_IGNORE);
    aixelerator_service::utils::writeP2PTimelineEvent(
        step, world_rank_, workgroup_rank_, true, "ml_step_end");
#ifdef SCOREP
    SCOREP_USER_REGION_END(pipelinedExchangeHandle)
#endif
}


template class CollectiveCommunication<float>;
template class CollectiveCommunication<double>;
