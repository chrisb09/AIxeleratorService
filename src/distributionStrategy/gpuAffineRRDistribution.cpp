#include "distributionStrategy/gpuAffineRRDistribution.h"

#include "utils/deviceCount.h"

#include <iostream>
#include <numeric>
#include <dlfcn.h>

#include <vector>
#include <map>
#include <iostream>
#include <limits.h>
#include <unistd.h>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#include <fstream>
#include <string>

#include <sstream>
#include <array>
#include <cstdio>
#include <algorithm>

GPUAffineRRDistribution::GPUAffineRRDistribution(MPI_Comm app_comm) 
{
    app_comm_ = app_comm;
    work_group_comm_ = app_comm_;
    /*
     * if AIxeleratorService is used together with PhyDLL or MLLIB in an MPMD 
     * run (i.e. app_comm_ will be different from MPI_COMM_WORLD), then we do
     * not want to use the GPU but leave it for PhyDLL or MLLIB.
     */
    if (app_comm_ == MPI_COMM_WORLD) {
        createWorkgroups();
    }
    else{
        workgroup_size_ = 0;
        num_devices_total_ = 0;
        is_gpu_controller_ = false;
    }
}

GPUAffineRRDistribution::~GPUAffineRRDistribution()
{

}

void GPUAffineRRDistribution::createWorkgroups()
{
    int my_rank, num_procs;
    int err;
    MPI_Comm_rank(app_comm_, &my_rank);
    MPI_Comm_size(app_comm_, &num_procs);
    my_rank_ = my_rank;

    // figure out our local node rank
    MPI_Comm node_communicator;
    MPI_Comm_split_type(app_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_communicator);
    int node_rank, node_size;
    MPI_Comm_rank(node_communicator, &node_rank);
    MPI_Comm_size(node_communicator, &node_size);
    MPI_Comm_free(&node_communicator);

    std::cout << "Rank " << my_rank << "/" << num_procs << " on its local machine is " << node_rank << "/" << node_size << std::endl; 

    int rank_numa_node = getRankNUMANode();
    std::vector<GPUInfo> gpus = discoverGPUs();

    std::cout << "Rank " << my_rank << " is bound to NUMA node " << rank_numa_node << std::endl;
    for (auto &gpu : gpus) {
        std::cout << "Rank " << my_rank << " GPU " << gpu.index << " is attached to NUMA node " << gpu.numa_node << std::endl;
    }


    // Default: not a GPU controller
    is_gpu_controller_ = false;
    my_gpu_device_ = -1;

    /*
     * Assign GPU controllers: lowest rank per NUMA node where a GPU lives.
     *
     * The selection must not run a different number of collectives on ranks
     * that see a different number of GPUs: in heterogeneous (CPU+GPU) runs the
     * CPU-only ranks see zero GPUs, so the per-GPU Allreduce of the original
     * implementation desynchronised the communicator. Instead, gather each
     * rank's host identity, NUMA node and local GPU list once, and derive the
     * controllers identically on every rank.
     */
    char my_processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len = 0;
    MPI_Get_processor_name(my_processor_name, &processor_name_len);
    std::vector<char> all_processor_names(
        static_cast<size_t>(num_procs) * MPI_MAX_PROCESSOR_NAME, '\0');
    MPI_Allgather(my_processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  all_processor_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, app_comm_);
    std::vector<int> all_host(num_procs, -1);
    std::vector<std::string> host_names;
    for (int rank = 0; rank < num_procs; ++rank) {
        const std::string name(
            all_processor_names.data() + static_cast<size_t>(rank) * MPI_MAX_PROCESSOR_NAME);
        int host = -1;
        for (int i = 0; i < static_cast<int>(host_names.size()); ++i) {
            if (host_names[i] == name) {
                host = i;
                break;
            }
        }
        if (host < 0) {
            host = static_cast<int>(host_names.size());
            host_names.push_back(name);
        }
        all_host[rank] = host;
    }

    std::vector<int> all_numa(num_procs, -1);
    MPI_Allgather(&rank_numa_node, 1, MPI_INT, all_numa.data(), 1, MPI_INT, app_comm_);

    std::vector<int> gpu_counts(num_procs, 0);
    int local_gpu_count = static_cast<int>(gpus.size());
    MPI_Allgather(&local_gpu_count, 1, MPI_INT, gpu_counts.data(), 1, MPI_INT, app_comm_);
    int max_gpus = 0;
    for (int count : gpu_counts) {
        max_gpus = std::max(max_gpus, count);
    }

    if (max_gpus > 0) {
        std::vector<int> send_packed(static_cast<size_t>(max_gpus) * 2, -1);
        for (int i = 0; i < local_gpu_count; ++i) {
            send_packed[static_cast<size_t>(i) * 2] = gpus[i].index;
            send_packed[static_cast<size_t>(i) * 2 + 1] = gpus[i].numa_node;
        }
        std::vector<int> all_packed(static_cast<size_t>(num_procs) * max_gpus * 2, -1);
        MPI_Allgather(send_packed.data(), max_gpus * 2, MPI_INT,
                      all_packed.data(), max_gpus * 2, MPI_INT, app_comm_);

        for (int owner = 0; owner < num_procs; ++owner) {
            for (int i = 0; i < gpu_counts[owner]; ++i) {
                const size_t offset = (static_cast<size_t>(owner) * max_gpus + i) * 2;
                const int gpu_index = all_packed[offset];
                const int gpu_numa = all_packed[offset + 1];
                if (gpu_numa < 0) {
                    continue;
                }
                int min_rank_on_node = INT_MAX;
                for (int rank = 0; rank < num_procs; ++rank) {
                    if (all_host[rank] == all_host[owner] && all_numa[rank] == gpu_numa) {
                        min_rank_on_node = std::min(min_rank_on_node, rank);
                    }
                }
                if (my_rank == min_rank_on_node) {
                    is_gpu_controller_ = true;
                    my_gpu_device_ = gpu_index;
                    std::cout << "Rank " << my_rank << " is GPU controller for GPU "
                              << gpu_index << " on NUMA node " << gpu_numa << std::endl;
                }
            }
        }
    }

    // figure out the total number of GPU controllers
    num_devices_total_ = 0;
    int my_num_devices = is_gpu_controller_ ? 1 : 0;
    MPI_Allreduce(&my_num_devices, &num_devices_total_, 1, MPI_INT, MPI_SUM, app_comm_);
    std::cout << "Rank " << my_rank << "/" << num_procs << " knows that there is a total of " << num_devices_total_ << " GPUs across all systems" << std::endl;

    if ( num_devices_total_ > 0)
    {
        // combine controllers (and workers) into separate communicators, to enumerate them
        MPI_Comm work_type_comm;
        MPI_Comm_split(app_comm_, my_num_devices, my_rank, &work_type_comm);
        int work_type_rank, work_type_size;
        MPI_Comm_rank(work_type_comm, &work_type_rank);
        MPI_Comm_size(work_type_comm, &work_type_size);
        MPI_Comm_free(&work_type_comm);

        // round robin assignment of data to a gpu, making sure the gpu master is rank 0
        int color = work_type_rank % num_devices_total_;
        int order = is_gpu_controller_ ? 0 : my_rank + num_devices_total_;

#ifdef WITH_NUMA_LOCAL_GROUPS
        /*
         * NUMA-local grouping: keep the GPU-affine controller selection, but
         * instead of a node-wide round robin, assign every worker to the
         * controller whose NUMA node is closest (ACPI SLIT distance), keeping
         * the group sizes balanced. On a 96c4g node this yields contiguous
         * 24-rank groups made of the NUMA domains adjacent to the controller's
         * GPU instead of a workgroup spread over all NUMA domains.
         *
         * On multi-node runs the SLIT matrix is only valid within one host, so
         * ranks on a different host get a large distance and are balanced
         * across controllers instead of being treated as NUMA-local.
         */
        {
            std::vector<int> all_numa(num_procs, -1);
            std::vector<int> all_is_controller(num_procs, 0);
            MPI_Allgather(&rank_numa_node, 1, MPI_INT, all_numa.data(), 1, MPI_INT, app_comm_);
            MPI_Allgather(&my_num_devices, 1, MPI_INT, all_is_controller.data(), 1, MPI_INT, app_comm_);

            char my_processor_name[MPI_MAX_PROCESSOR_NAME];
            int processor_name_len = 0;
            MPI_Get_processor_name(my_processor_name, &processor_name_len);
            std::vector<char> all_processor_names(
                static_cast<size_t>(num_procs) * MPI_MAX_PROCESSOR_NAME, '\0');
            MPI_Allgather(my_processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                          all_processor_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, app_comm_);
            std::vector<int> all_host(num_procs, -1);
            std::vector<std::string> host_names;
            for (int rank = 0; rank < num_procs; ++rank) {
                const std::string name(
                    all_processor_names.data() + static_cast<size_t>(rank) * MPI_MAX_PROCESSOR_NAME);
                int host = -1;
                for (int i = 0; i < static_cast<int>(host_names.size()); ++i) {
                    if (host_names[i] == name) {
                        host = i;
                        break;
                    }
                }
                if (host < 0) {
                    host = static_cast<int>(host_names.size());
                    host_names.push_back(name);
                }
                all_host[rank] = host;
            }
            const bool multi_host = host_names.size() > 1;

            std::vector<int> controller_ranks;
            std::vector<int> controller_numa;
            for (int rank = 0; rank < num_procs; ++rank) {
                if (all_is_controller[rank]) {
                    controller_ranks.push_back(rank);
                    controller_numa.push_back(all_numa[rank]);
                }
            }

            if (static_cast<int>(controller_ranks.size()) != num_devices_total_) {
                std::cerr << "NUMA-local grouping unavailable (found " << controller_ranks.size()
                          << " controllers for " << num_devices_total_
                          << " GPUs); falling back to round robin." << std::endl;
            } else {
                const int base_size = num_procs / num_devices_total_;
                const int remainder = num_procs % num_devices_total_;
                std::vector<int> capacity(num_devices_total_, base_size);
                for (int i = 0; i < remainder; ++i) {
                    ++capacity[i];
                }
                std::vector<int> assigned_group(num_procs, -1);
                std::vector<int> members(num_devices_total_, 0);
                for (int i = 0; i < num_devices_total_; ++i) {
                    assigned_group[controller_ranks[i]] = i;
                    members[i] = 1;
                }

                const int cross_host_distance = 1000000;
                for (int rank = 0; rank < num_procs; ++rank) {
                    if (assigned_group[rank] >= 0) {
                        continue;
                    }
                    int best_group = -1;
                    int best_distance = INT_MAX;
                    int best_members = INT_MAX;
                    for (int i = 0; i < num_devices_total_; ++i) {
                        if (members[i] >= capacity[i]) {
                            continue;
                        }
                        int distance = INT_MAX;
                        if (multi_host && all_host[rank] != all_host[controller_ranks[i]]) {
                            distance = cross_host_distance;
                        } else if (all_numa[rank] >= 0 && controller_numa[i] >= 0) {
                            const int node_distance = numa_distance(all_numa[rank], controller_numa[i]);
                            if (node_distance > 0) {
                                distance = node_distance;
                            }
                        }
                        // Single-host keeps the original distance-only tie-break so
                        // existing single-node results stay reproducible. Multi-host
                        // additionally balances member counts, otherwise all remote
                        // ranks would pile onto the first controller.
                        const bool is_better = multi_host
                            ? (distance < best_distance ||
                               (distance == best_distance && members[i] < best_members))
                            : (distance < best_distance);
                        if (is_better) {
                            best_distance = distance;
                            best_members = members[i];
                            best_group = i;
                        }
                    }
                    if (best_group < 0) {
                        best_group = rank % num_devices_total_;
                    }
                    assigned_group[rank] = best_group;
                    ++members[best_group];
                }

                color = assigned_group[my_rank];
                std::cout << "Rank " << my_rank << "/" << num_procs
                          << " NUMA-local assignment: rank NUMA " << rank_numa_node
                          << " host " << all_host[my_rank] << "/" << host_names.size()
                          << " -> group " << color
                          << " (controller rank " << controller_ranks[color] << ")" << std::endl;
            }
        }
#endif

        std::cout << "Rank " << my_rank << "/" << num_procs << " will be in group " << color << " order " << order << std::endl;

        // initialize the work group communicator
        err = MPI_Comm_split(app_comm_, color, order, &work_group_comm_);
        if ( err != MPI_SUCCESS )
        {
            std::cout << "Rank " << my_rank << ": Error when splitting workgroup communicator " << work_group_comm_  << std::endl;
        }
        int work_group_rank, work_group_size;
        MPI_Comm_rank(work_group_comm_, &work_group_rank);
        MPI_Comm_size(work_group_comm_, &work_group_size);
        workgroup_size_ = work_group_size;
        std::cout << "Rank " << my_rank << "/" << num_procs << " got work group rank id " << work_group_rank << "/" << work_group_size << std::endl;
    }
    else
    {
        workgroup_size_ = 0;
        work_group_comm_ = app_comm_;
    } 
}

int GPUAffineRRDistribution::getRankNUMANode() {
    // Get the current CPU the thread is running on
    int cpu = sched_getcpu();
    if (cpu < 0) {
        perror("sched_getcpu failed");
        return -1;
    }

    // Get the NUMA node of that CPU
    int node = numa_node_of_cpu(cpu);
    if (node < 0) {
        perror("numa_node_of_cpu failed");
        return -1;
    }

    return node;
}

std::vector<GPUInfo> GPUAffineRRDistribution::discoverGPUs() {
    std::vector<GPUInfo> gpus;

    int count = aixelerator_service::utils::deviceCount();
    std::cout << "Found " << count << " devices" << std::endl;
    if (count <= 0) {
        std::cerr << "No devices found.\n";
        return gpus;
    }

    void* cuda_rt = dlopen("libcudart.so", RTLD_LAZY);
    void* veda_rt = dlopen("libveda.so.0", RTLD_LAZY);

    if (veda_rt != nullptr) {
        // VEDA devices: currently not implemented
        std::cerr << "VEDA runtime detected; NUMA node detection not implemented.\n";
        dlclose(veda_rt);
        return gpus;
    }

    if (cuda_rt == nullptr) {
        std::cerr << "Could not open libcudart.so\n";
        return gpus;
    }

    using cudaDeviceGetPCIBusId_t = int(*)(char*, int, int);
    auto cudaDeviceGetPCIBusId = (cudaDeviceGetPCIBusId_t)dlsym(cuda_rt, "cudaDeviceGetPCIBusId");
    if (!cudaDeviceGetPCIBusId) {
        std::cerr << "Could not find cudaDeviceGetPCIBusId\n";
        dlclose(cuda_rt);
        return gpus;
    }

    // TODO: hwloc deployments built without PCI/GPU support produce no usable
    // GPU NUMA mapping here. Add a fallback that reads the NUMA node directly
    // from /sys/bus/pci/devices/<pci_id>/numa_node instead of parsing lstopo.
    // Run lstopo and capture its text output
    FILE* pipe = popen("lstopo", "r");
    if (!pipe) {
        std::cerr << "Failed to run lstopo\n";
        dlclose(cuda_rt);
        return gpus;
    }

    std::vector<std::string> topo_lines;
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        topo_lines.emplace_back(buffer);
    }
    pclose(pipe);
    
    // helper: lowercase a string
    auto toLower = [](std::string s) -> std::string {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    };

    // helper: extract the 'tail' of a PCI id: "0000:65:00.0" -> "65:00.0", "65:00.0" -> "65:00.0"
    auto extract_pci_tail = [](const std::string &s) -> std::string {
        // trim
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\r");
        std::string t = s.substr(start, end - start + 1);

        size_t pos_dot = t.find_last_of('.');
        if (pos_dot == std::string::npos) {
            // no dot -> fallback to full trimmed string
            return t;
        }
        // find the ':' before the dot (device:function separator) and the previous ':' (domain separator)
        size_t pos_colon_before = t.rfind(':', pos_dot);
        if (pos_colon_before == std::string::npos) return t;
        size_t pos_colon_prev = (pos_colon_before == 0) ? std::string::npos : t.rfind(':', pos_colon_before - 1);
        size_t start_idx = (pos_colon_prev == std::string::npos) ? 0 : (pos_colon_prev + 1);
        return t.substr(start_idx);
    };

    // For each GPU (via CUDA), find its PCI ID and then find the nearest NUMA node ABOVE it in lstopo output
    for (int i = 0; i < count; ++i) {
        char pciBusId[64] = {0};
        if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), i) != 0) {
            std::cerr << "Failed to get PCI Bus ID for GPU " << i << "\n";
            gpus.push_back({i, -1});
            continue;
        }
        std::string pciIdOrig = pciBusId;
        std::string pciTail = extract_pci_tail(pciIdOrig);
        std::string pciTailLower = toLower(pciTail);
        std::string pciOrigLower = toLower(pciIdOrig);

        std::cout << "pciBusId " << pciIdOrig << " of gpu " << i << std::endl;

        int found_index = -1;
        // find the first line containing either the full pci id or the tail
        for (int l = 0; l < (int)topo_lines.size(); ++l) {
            std::string low = toLower(topo_lines[l]);
            if (low.find(pciTailLower) != std::string::npos || low.find(pciOrigLower) != std::string::npos) {
                found_index = l;
                break;
            }
        }

        int numa_node = -1;
        if (found_index != -1) {
            // NUMA node must be above: search upward (towards beginning of file)
            for (int j = found_index; j >= 0; --j) {
                std::string low = toLower(topo_lines[j]);
                size_t pos = low.find("numanode");
                if (pos != std::string::npos) {
                    int node = -1;
                    const char* c = low.c_str() + pos;
                    // try formats: "NUMANode L#%d" or "NUMANode %d"
                    if (sscanf(c, "numanode l#%d", &node) == 1 || sscanf(c, "numanode %d", &node) == 1) {
                        numa_node = node;
                        break;
                    }
                }
                // keep searching upward until start of the output (no artificial small range)
            }
        } else {
            std::cerr << "Could not find PCI ID " << pciIdOrig << " in lstopo output\n";
        }

        std::cout << "Normalized pciBusId " << pciTail << " of gpu " << i << " -> NUMA node " << numa_node << std::endl;
        gpus.push_back({i, numa_node});
    }
    
    dlclose(cuda_rt);
    return gpus;
}
