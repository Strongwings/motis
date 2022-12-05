#include "motis/tripbased/gpu/gpu_postprocessing.h"

#include "motis/tripbased/error.h"

#ifdef MOTIS_CUDA

namespace motis::tripbased {

gpu_postproc_result::gpu_postproc_result(
    gpu_search_results const& results,
    std::vector<std::vector<destination_arrival>> const& dest_arrivals) {
  queue_results_.resize(results.gpu_final_queues_.size());
  for (auto i = 0; i < results.gpu_final_queues_.size(); ++i) {
    for (auto qe : results.gpu_final_queues_[i]) {
      queue_results_[i].emplace_back(queue_entry(qe.trip_, qe.from_stop_index_,
                                                 qe.to_stop_index_,
                                                 qe.previous_trip_segment_));
    }
  }

  if (results.gpu_result_journeys_.size() == 0) {
    return;
  }

  for (auto i = 0; i < results.gpu_result_journeys_.size(); ++i) {
    auto const& res = results.gpu_result_journeys_[i];
    if (results.gpu_is_dominated_[i] == 0
        && res.arrival_time_ - res.start_time_ <= MAX_TRAVEL_TIME) {
      if (!dest_arrivals[res.destination_arrival_.line_id_].empty()) {
        auto dest_arr = 0;
        if (dest_arrivals[res.destination_arrival_.line_id_].size() > 1) {
          auto const& da = dest_arrivals[res.destination_arrival_.line_id_];
          for(auto j = 0; j < da.size(); ++j) {
            if (da[j].stop_index_ == res.destination_arrival_.stop_index_
                && da[j].footpath_.from_stop_ == res.destination_arrival_.fp_from_station_id_
                && da[j].footpath_.to_stop_ == res.destination_arrival_.fp_to_station_id_
                && da[j].footpath_.duration_ == res.destination_arrival_.fp_duration_) {
              dest_arr = j;
              break;
            }
          }
        }

        auto jny = tb_journey(search_dir::FWD, res.start_time_,
                              res.arrival_time_, res.transfers_,
                              res.transports_, res.destination_station_,
                              &dest_arrivals[res.destination_arrival_.line_id_][dest_arr],
                              res.final_queue_entry_);
        journey_results_.emplace_back(jny);
      } else {
        throw std::system_error(error::internal_error);
      }
    }
  }
}

} // namespace motis::tripbased

#endif