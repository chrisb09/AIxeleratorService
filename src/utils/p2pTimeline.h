#ifndef AIXELERATORSERVICE_UTILS_P2PTIMELINE_H_
#define AIXELERATORSERVICE_UTILS_P2PTIMELINE_H_

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

namespace aixelerator_service::utils {
inline double p2pTimelineTime()
{
    const char* offset_text = std::getenv("AIX_P2P_CLOCK_OFFSET_SECONDS");
    const double offset = offset_text ? std::strtod(offset_text, nullptr) : 0.0;
    return MPI_Wtime() + offset;
}

struct P2PTimelineEvent {
    uint64_t step;
    double time_s;
    int world_rank;
    int workgroup_rank;
    bool is_controller;
    std::string event;
    int peer_workgroup_rank;
    int range_first_rank;
    int range_end_rank;
    int64_t sample_start;
    int64_t sample_count;
};

struct P2PTimelineThreadContext {
    bool active = false;
    int world_rank = -1;
    int workgroup_rank = -1;
    bool is_controller = false;
    int range_first_rank = -1;
    int range_end_rank = -1;
};

inline P2PTimelineThreadContext& p2pTimelineThreadContext()
{
    static thread_local P2PTimelineThreadContext context;
    return context;
}

class P2PTimelineThreadContextGuard {
public:
    P2PTimelineThreadContextGuard(int world_rank, int workgroup_rank, bool is_controller,
                                  int range_first_rank, int range_end_rank)
        : previous_(p2pTimelineThreadContext())
    {
        p2pTimelineThreadContext() = {
            true, world_rank, workgroup_rank, is_controller, range_first_rank, range_end_rank};
    }

    ~P2PTimelineThreadContextGuard()
    {
        p2pTimelineThreadContext() = previous_;
    }

private:
    P2PTimelineThreadContext previous_;
};

inline std::vector<P2PTimelineEvent>& p2pTimelineEvents()
{
    static std::vector<P2PTimelineEvent> events;
    return events;
}

inline std::string& p2pTimelineDirectory()
{
    static std::string directory;
    return directory;
}

inline std::mutex& p2pTimelineMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline void writeP2PTimelineEvent(uint64_t step, int world_rank, int workgroup_rank, bool is_controller,
                                  const char* event, int peer_workgroup_rank = -1,
                                  int range_first_rank = -1, int range_end_rank = -1,
                                  int64_t sample_start = -1, int64_t sample_count = -1)
{
    const char* directory = std::getenv("AIX_P2P_TIMELINE_DIR");
    if (!directory || directory[0] == '\0') {
        return;
    }
    const double time_s = p2pTimelineTime();
    std::lock_guard<std::mutex> lock(p2pTimelineMutex());
    if (p2pTimelineDirectory().empty()) {
        p2pTimelineDirectory() = directory;
    }
    p2pTimelineEvents().push_back({
        step, time_s, world_rank, workgroup_rank, is_controller, event,
        peer_workgroup_rank, range_first_rank, range_end_rank, sample_start, sample_count});
}

inline void writeP2PTimelineEventAt(uint64_t step, double time_s, int world_rank, int workgroup_rank,
                                    bool is_controller, const char* event, int peer_workgroup_rank = -1,
                                    int range_first_rank = -1, int range_end_rank = -1,
                                    int64_t sample_start = -1, int64_t sample_count = -1)
{
    const char* directory = std::getenv("AIX_P2P_TIMELINE_DIR");
    if (!directory || directory[0] == '\0') {
        return;
    }
    std::lock_guard<std::mutex> lock(p2pTimelineMutex());
    if (p2pTimelineDirectory().empty()) {
        p2pTimelineDirectory() = directory;
    }
    p2pTimelineEvents().push_back({
        step, time_s, world_rank, workgroup_rank, is_controller, event,
        peer_workgroup_rank, range_first_rank, range_end_rank, sample_start, sample_count});
}

inline void flushP2PTimelineEvents()
{
    std::string directory;
    std::vector<P2PTimelineEvent> events;
    {
        std::lock_guard<std::mutex> lock(p2pTimelineMutex());
        directory = p2pTimelineDirectory();
        events.swap(p2pTimelineEvents());
    }
    if (directory.empty() || events.empty()) {
        return;
    }

    std::filesystem::create_directories(directory);
    const std::filesystem::path path = std::filesystem::path(directory) /
        ("aix_p2p_timeline_rank_" + std::to_string(events.front().world_rank) + ".csv");
    const bool write_header = !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::app);
    if (write_header) {
        stream << "step,time_s,world_rank,workgroup_rank,is_controller,event,peer_workgroup_rank,"
               << "range_first_rank,range_end_rank,sample_start,sample_count\n";
    }
    for (const auto& event : events) {
        stream << event.step << ',' << std::setprecision(17) << event.time_s << ','
               << event.world_rank << ',' << event.workgroup_rank << ',' << (event.is_controller ? 1 : 0) << ','
               << event.event << ',' << event.peer_workgroup_rank << ',' << event.range_first_rank << ','
               << event.range_end_rank << ',' << event.sample_start << ',' << event.sample_count << '\n';
    }
}

inline int p2pTimelineEnvironmentInt(const char* name, int fallback = -1)
{
    const char* text = std::getenv(name);
    return text ? std::atoi(text) : fallback;
}

inline int64_t p2pTimelineEnvironmentInt64(const char* name, int64_t fallback = -1)
{
    const char* text = std::getenv(name);
    return text ? std::strtoll(text, nullptr, 10) : fallback;
}

inline void writeP2PTorchForwardEvent(const char* event, int64_t sample_start, int64_t sample_count)
{
    const int world_rank = p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_WORLD_RANK");
    if (world_rank < 0) {
        return;
    }
    writeP2PTimelineEvent(
        static_cast<uint64_t>(p2pTimelineEnvironmentInt64("AIX_P2P_TIMELINE_STEP", 0)),
        world_rank,
        p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_WORKGROUP_RANK"),
        true,
        event,
        -1,
        p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_RANGE_FIRST_RANK"),
        p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_RANGE_END_RANK"),
        sample_start,
        sample_count);
}

inline void writeP2PTorchForwardEventAt(double time_s, const char* event,
                                        int64_t sample_start, int64_t sample_count)
{
    const auto& context = p2pTimelineThreadContext();
    const int world_rank = context.active ? context.world_rank :
        p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_WORLD_RANK");
    if (world_rank < 0) {
        return;
    }
    const char* directory = std::getenv("AIX_P2P_TIMELINE_DIR");
    if (!directory || directory[0] == '\0') {
        return;
    }
    std::lock_guard<std::mutex> lock(p2pTimelineMutex());
    if (p2pTimelineDirectory().empty()) {
        p2pTimelineDirectory() = directory;
    }
    p2pTimelineEvents().push_back({
        static_cast<uint64_t>(p2pTimelineEnvironmentInt64("AIX_P2P_TIMELINE_STEP", 0)),
        time_s,
        world_rank,
        context.active ? context.workgroup_rank :
            p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_WORKGROUP_RANK"),
        context.active ? context.is_controller : true,
        event,
        -1,
        context.active ? context.range_first_rank :
            p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_RANGE_FIRST_RANK"),
        context.active ? context.range_end_rank :
            p2pTimelineEnvironmentInt("AIX_P2P_TIMELINE_RANGE_END_RANK"),
        sample_start,
        sample_count});
}
} // namespace aixelerator_service::utils

#endif
