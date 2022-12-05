#include "motis/tripbased/gpu/gpu_timetable.h"

#ifdef MOTIS_CUDA

namespace motis::tripbased {

gpu_timetable::gpu_timetable(fws_multimap<motis::time> arrival_times,
                             mcd::vector<stop_idx_t> line_stop_count,
                             nested_fws_multimap<tb_transfer> transfers,
                             mcd::vector<line_id> trip_to_line,
                             uint64_t const& trip_count) {

  queue_size_ = 0;
  for (auto i = 0; i < MAX_TRANSFERS; ++i) {
    queue_size_ += 4000000;
    queue_index_.emplace_back(4000000 * i);
  }

  gpu_fws_multimap_arrival_times gpu_arrival_times;
  gpu_arrival_times.data_ = arrival_times.data_.data();
  gpu_arrival_times.index_ = arrival_times.index_.data();
  std::size_t at_index_size = arrival_times.index_.size();
  gpu_arrival_times.index_size_ = &at_index_size;

  std::vector<gpu_tb_transfer> gpu_tb_transfers;
  for (auto const& trans : transfers.data_) {
    gpu_tb_transfers.emplace_back(gpu_tb_transfer{trans.to_trip_, trans.to_stop_idx_});
  }
  gpu_nested_fws_multimap_transfers gpu_transfers;
  gpu_transfers.data_ = gpu_tb_transfers.data();
  gpu_transfers.base_index_ = (std::size_t*)transfers.base_index_.data();
  std::size_t t_base_index_size = transfers.base_index_.size();
  gpu_transfers.base_index_size_ = &t_base_index_size;
  gpu_transfers.index_ = transfers.index_.data();
  std::size_t t_index_size = transfers.index_.size();
  gpu_transfers.index_size_ = &t_index_size;

  ptrs_ = allocate_and_copy_on_device(gpu_arrival_times,
                                      line_stop_count.data(),
                                      line_stop_count.size(),
                                      gpu_transfers,
                                      trip_to_line.data(),
                                      trip_to_line.size(),
                                      trip_count,
                                      queue_size_,
                                      queue_index_,
                                      result_set_alloc_num_);
}

gpu_device_query_pointers create_query_pointers(
    gpu_device_pointers const& pointers,
    std::vector<std::vector<destination_arrival>> const& dest_arrivals,
    uint64_t const& trip_count,
    std::vector<queue_entry> const& initial_queue,
    std::size_t const& result_set_alloc_num) {

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

  std::vector<gpu_queue_entry> gpu_initial_queue;
  for(auto const& qe : initial_queue) {
    gpu_initial_queue.emplace_back(
        gpu_queue_entry{qe.trip_, qe.from_stop_index_, qe.to_stop_index_,
                        qe.previous_trip_segment_});
  }

  return allocate_and_copy_on_device_query(pointers,
                                           gpu_dest_arrivals,
                                           gpu_initial_queue,
                                           trip_count,
                                           result_set_alloc_num);

}

} // namespace motis::tripbased

#endif