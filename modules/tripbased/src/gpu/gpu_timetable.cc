#include "motis/tripbased/gpu/gpu_timetable.h"

#ifdef MOTIS_CUDA

#include "motis/tripbased/gpu/gpu_tripbased.h"

namespace motis::tripbased {

gpu_timetable::gpu_timetable(std::vector<std::vector<destination_arrival>> dest_arrivals,
                             fws_multimap<motis::time> arrival_times,
                             time total_earliest_arrival,
                             mcd::vector<stop_idx_t> line_stop_count,
                             nested_fws_multimap<tb_transfer> transfers) {

  std::vector<std::vector<gpu_dest_arrival>> gpu_dest_arrivals;
  for(auto const &dest_arrs : dest_arrivals) {
    std::vector<gpu_dest_arrival> gpu_dest_arrs;
    for(auto const &dest_arr : dest_arrs) {
      gpu_dest_arrs.emplace_back(
          gpu_dest_arrival{
              dest_arr.line_,
              dest_arr.stop_index_,
              dest_arr.footpath_.from_stop_,
              dest_arr.footpath_.to_stop_,
              dest_arr.footpath_.duration_});
    }
    gpu_dest_arrivals.emplace_back(gpu_dest_arrs);
  }

  gpu_fws_multimap_arrival_times gpu_arrival_times;
  gpu_arrival_times.data_ = arrival_times.data_.data();
  gpu_arrival_times.index_ = arrival_times.index_.data();
  gpu_arrival_times.index_size_ = arrival_times.index_.size();

  std::vector<gpu_tb_transfer> gpu_tb_transfers;
  for(auto trans : transfers.data_) {
    gpu_tb_transfers.emplace_back(gpu_tb_transfer{trans.to_trip_, trans.to_stop_idx_});
  }
  gpu_nested_fws_multimap_transfers gpu_transfers;
  gpu_transfers.data_ = gpu_tb_transfers.data();
  gpu_transfers.base_index_ = (std::size_t*)transfers.base_index_.data();
  gpu_transfers.base_index_size_ = transfers.base_index_.size();
  gpu_transfers.index_ = transfers.index_.data();
  gpu_transfers.index_size_ = transfers.index_.size();

  // TODO(sarah)

  ptrs_ = allocate_and_copy_on_device(gpu_dest_arrivals,
                                      gpu_arrival_times,
                                      total_earliest_arrival,
                                      line_stop_count.data(),
                                      line_stop_count.size(),
                                      gpu_transfers);
}

gpu_timetable::~gpu_timetable() {
  free_on_device(ptrs_);
}

} // namespace motis::tripbased
#endif