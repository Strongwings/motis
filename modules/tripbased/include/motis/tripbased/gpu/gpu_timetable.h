#pragma once

#include <vector>

#include "motis/tripbased/tb_journey.h"

#ifdef MOTIS_CUDA
#include "motis/tripbased/gpu/gpu_tripbased.h"
#endif

namespace motis::tripbased {
struct gpu_timetable {

#ifdef MOTIS_CUDA
  gpu_device_ptrs ptrs_;
#endif

  gpu_timetable(std::vector<std::vector<destination_arrival>> dest_arrivals,
                fws_multimap<motis::time> arrival_times,
                time total_earliest_arrival,
                mcd::vector<stop_idx_t> line_stop_count,
                nested_fws_multimap<tb_transfer> transfers);
  ~gpu_timetable();
};
} // namespace motis::tripbased