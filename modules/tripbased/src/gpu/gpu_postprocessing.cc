#include "motis/tripbased/gpu/gpu_postprocessing.h"

#ifdef MOTIS_CUDA

namespace motis::tripbased {

gpu_postproc_result::gpu_postproc_result(
    gpu_search_results results,
    std::vector<std::vector<destination_arrival>>& dest_arrivals) {
  queue_results_.resize(results.gpu_final_queues_.size());
  for (auto i = 0; i < results.gpu_final_queues_.size(); ++i) {
    for (auto qe : results.gpu_final_queues_[i]) {
      queue_results_[i].emplace_back(queue_entry(qe.trip_, qe.from_stop_index_,
                                                 qe.to_stop_index_,
                                                 qe.previous_trip_segment_));
    }
  }

  for (auto i = 0; i < results.gpu_result_journeys_.size(); ++i) {
    auto res = results.gpu_result_journeys_[i];
    if (results.gpu_is_dominated_[i] == 0) {
      if (!dest_arrivals[res.destination_arrival_.line_id_].empty()) {
        // TODO(sarah)
        if (dest_arrivals[res.destination_arrival_.line_id_].size() > 1) {
          auto test = dest_arrivals[res.destination_arrival_.line_id_];
          std::cout << "test2" << std::endl;
        }
        auto& test2 = dest_arrivals[res.destination_arrival_.line_id_][0];
        auto test = tb_journey(search_dir::FWD, res.start_time_,
                               res.arrival_time_, res.transfers_,
                               res.transports_, res.destination_station_,
                               &test2, res.final_queue_entry_);
        journey_results_.emplace_back(test);
      } else {
        std::cout << "test" << std::endl;
        assert(false);
      }
    }
  }
}

} // namespace motis::tripbased

#endif