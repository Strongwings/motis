#pragma once

#include <vector>

#include "motis/tripbased/tb_journey.h"

#ifdef MOTIS_CUDA
#include "motis/tripbased/gpu/gpu_tripbased.h"

namespace motis::tripbased {
struct gpu_timetable {

  gpu_device_pointers ptrs_;

  gpu_timetable(std::vector<std::vector<destination_arrival>> dest_arrivals,
                fws_multimap<motis::time> arrival_times,
                time total_earliest_arrival,
                mcd::vector<stop_idx_t> line_stop_count,
                nested_fws_multimap<tb_transfer> transfers,
                mcd::vector<line_id> trip_to_line,
                time start_time,
                std::vector<stop_idx_t> first_reachable_stop,
                uint64_t trip_count,
                std::vector<queue_entry> initial_queue,
                unsigned max_transfers);
  ~gpu_timetable();

};

} // namespace motis::tripbased

#endif