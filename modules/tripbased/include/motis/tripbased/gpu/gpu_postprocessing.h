#pragma once

#include <vector>

#include "motis/tripbased/tb_journey.h"

#ifdef MOTIS_CUDA
#include "motis/tripbased/gpu/gpu_tripbased.h"

namespace motis::tripbased {

struct gpu_postproc_result {
  std::vector<std::vector<queue_entry>> queue_results_;
  std::vector<tb_journey> journey_results_;

  gpu_postproc_result(
      gpu_search_results const& results,
      std::vector<std::vector<destination_arrival>> const& dest_arrivals);
};

} // namespace motis::tripbased

#endif